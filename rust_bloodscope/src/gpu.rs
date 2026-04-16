//! GPU (wgpu) backend for bloodscope enumeration.
//!
//! Dispatches a WGSL compute shader that runs the full RNG + scope-sampling
//! pipeline, one thread per seed. Output is packed (ft_pct | bk_pct << 8) as
//! a u32 per seed. Runs in batches of configurable size to avoid TDR
//! timeouts and keep VRAM usage bounded.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Params {
    start_lo:    u32,
    start_hi:    u32,
    num_seeds:   u32,
    num_offsets: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct OffsetVec2 {
    x: f32,
    y: f32,
}

pub struct GpuRunner {
    pub device:        wgpu::Device,
    pub queue:         wgpu::Queue,
    pipeline:          wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    offsets_buf:       wgpu::Buffer,
    pattern_buf:       wgpu::Buffer,
    params_buf:        wgpu::Buffer,
    output_buf:        wgpu::Buffer,
    staging_buf:       wgpu::Buffer,
    bind_group:        wgpu::BindGroup,
    pub batch_size:    u32,
    num_offsets:       u32,
    pub adapter_name:  String,
}

impl GpuRunner {
    pub async fn new(
        offsets: &[(f32, f32)],
        combined_pattern: &[u8],
        batch_size: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        eprintln!("GPU: {adapter_name}");

        // Need a 256MB storage buffer for the max batch; leave headroom.
        let pattern_bytes = combined_pattern.len() as u64;      // 4 MB
        let offsets_bytes = (offsets.len() * 8) as u64;         // ~75 KB
        let output_bytes  = (batch_size as u64) * 4;            // 4B * batch

        let desc = wgpu::DeviceDescriptor {
            label: Some("bloodscope device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: (output_bytes
                    .max(pattern_bytes)
                    .max(offsets_bytes) as u32)
                    .max(256 << 20),
                max_buffer_size: output_bytes
                    .max(pattern_bytes)
                    .max(offsets_bytes)
                    .max(256 << 20),
                max_compute_invocations_per_workgroup: WORKGROUP_SIZE,
                max_compute_workgroup_size_x: WORKGROUP_SIZE,
                max_compute_workgroups_per_dimension: 65535,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = adapter
            .request_device(&desc)
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        // -- Shader + pipeline --
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloodscope shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloodscope cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_bloodscope"),
            compilation_options: Default::default(),
            cache: None,
        });

        // -- Static buffers: offsets (vec2<f32>) and combined pattern (u32) --
        let offsets_data: Vec<OffsetVec2> =
            offsets.iter().map(|(x, y)| OffsetVec2 { x: *x, y: *y }).collect();
        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets"),
            contents: bytemuck::cast_slice(&offsets_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Re-pack pattern bytes into u32 words.
        let mut words: Vec<u32> = Vec::with_capacity(combined_pattern.len() / 4);
        for chunk in combined_pattern.chunks_exact(4) {
            words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let pattern_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pattern"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // -- Params (uniform) ------------------------------------------------
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Output (device-side) + staging (host-readable) ------------------
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pattern_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
            offsets_buf,
            pattern_buf,
            params_buf,
            output_buf,
            staging_buf,
            bind_group,
            batch_size,
            num_offsets: offsets.len() as u32,
            adapter_name,
        })
    }

    /// Process `num_seeds` seeds starting at `start_seed` and return a
    /// packed u32 per seed: low byte = FT pct, second byte = Buckets pct.
    pub fn run_batch(&self, start_seed: u64, num_seeds: u32) -> Vec<u32> {
        assert!(num_seeds <= self.batch_size);

        // 1. Upload params (pass full 64-bit start seed as two u32s)
        let params = Params {
            start_lo:    start_seed as u32,
            start_hi:    (start_seed >> 32) as u32,
            num_seeds,
            num_offsets: self.num_offsets,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // 2. Dispatch (2D so we can exceed 65535 * 64 = ~4M threads per dim)
        let total_wg = num_seeds.div_ceil(WORKGROUP_SIZE);
        const MAX_DIM: u32 = 65535;
        let wg_x = total_wg.min(MAX_DIM);
        let wg_y = total_wg.div_ceil(wg_x);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bloodscope enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bloodscope pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 3. Copy output -> staging
        let bytes = (num_seeds as u64) * 4;
        enc.copy_buffer_to_buffer(&self.output_buf, 0, &self.staging_buf, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        // 4. Read back
        let slice = self.staging_buf.slice(0..bytes);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::PollType::Wait).ok();
        rx.recv().expect("map channel").expect("map failed");
        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        self.staging_buf.unmap();
        result
    }
}
