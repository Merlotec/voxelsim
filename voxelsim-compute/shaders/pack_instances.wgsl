struct CameraUniform {
    view_proj: mat4x4<f32>,
    pos: vec3<f32>,
}
var<push_constant> camera: CameraUniform;

// Noise uniform (matches NoiseBuffer layout)
struct Noise6Params {
  spatial_freq : vec3<f32>,
  seed_freq    : vec3<f32>,
  seed_vec     : vec3<f32>,
  lacunarity   : f32,
  gain         : f32,
  octaves      : u32,
  enabled      : u32,
}
@group(0) @binding(2) var<uniform> noise6 : Noise6Params;

struct CellInstance {
    position: vec3<i32>,
    value: u32,
};

struct CullParams {
    instance_count: u32,
    visible_count: u32,
    _padding: vec2<u32>,
};

struct KVOut { coord: vec3<i32>, value: u32 };
struct OutputBuffer { count: atomic<u32>, data: array<KVOut> };

@group(0) @binding(0) var<storage, read> culled_instances: array<CellInstance>;
@group(0) @binding(1) var<storage, read> cull_params: CullParams;
@group(0) @binding(3) var<storage, read_write> out_buf: OutputBuffer;

// ===== Noise helpers (copied from voxelcoord) =====
const F6 : f32 = (sqrt(7.0) - 1.0) / 6.0;
const G6 : f32 = (1.0 - 1.0 / sqrt(7.0)) / 6.0;
const ATTN6 : f32 = 0.5;

fn mix32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u; h *= 0x7feb352du;
    h ^= h >> 15u; h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn randSym(seed: u32) -> f32 {
    let r = mix32(seed);
    return f32(r) * (1.0 / 4294967295.0) * 2.0 - 1.0;
}

struct G6s { ga: vec3<f32>, gb: vec3<f32> }

fn hash6(i0:i32, i1:i32, i2:i32, i3:i32, i4:i32, i5:i32) -> u32 {
    var h : u32 =
        bitcast<u32>(i0) * 0x8da6b343u ^
        bitcast<u32>(i1) * 0xd8163841u ^
        bitcast<u32>(i2) * 0xcb1ab31fu ^
        bitcast<u32>(i3) * 0x165667b1u ^
        bitcast<u32>(i4) * 0x9e3779b9u ^
        bitcast<u32>(i5) * 0x85ebca6bu;
    return mix32(h);
}

fn grad6(h: u32) -> G6s {
    let g0 = randSym(h ^ 0x9e3779b9u);
    let g1 = randSym(h ^ 0x3c6ef372u);
    let g2 = randSym(h ^ 0xda3e39cbu);
    let g3 = randSym(h ^ 0xbb67ae85u);
    let g4 = randSym(h ^ 0x6a09e667u);
    let g5 = randSym(h ^ 0xa54ff53au);
    var ga = vec3<f32>(g0, g1, g2);
    var gb = vec3<f32>(g3, g4, g5);
    let inv = inverseSqrt(max(dot(ga,ga) + dot(gb,gb), 1e-6));
    ga *= inv; gb *= inv;
    return G6s(ga, gb);
}

fn snoise6(pos: vec3<f32>, seed: vec3<f32>, freq_scale: f32) -> f32 {
    var p : array<f32, 6>;
    p[0] = pos.x  * (noise6.spatial_freq.x * freq_scale);
    p[1] = pos.y  * (noise6.spatial_freq.y * freq_scale);
    p[2] = pos.z  * (noise6.spatial_freq.z * freq_scale);
    p[3] = seed.x * (noise6.seed_freq.x    * freq_scale);
    p[4] = seed.y * (noise6.seed_freq.y    * freq_scale);
    p[5] = seed.z * (noise6.seed_freq.z    * freq_scale);

    var sumP : f32 = 0.0;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { sumP += p[a]; }
    let s : f32 = sumP * F6;

    var i : array<i32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { i[a] = i32(floor(p[a] + s)); }

    var sumI : i32 = 0;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { sumI += i[a]; }
    let t : f32 = f32(sumI) * G6;

    var x0 : array<f32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { x0[a] = p[a] - f32(i[a]) + t; }

    var rank : array<u32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { rank[a] = 0u; }
    for (var a: u32 = 0u; a < 6u; a = a + 1u) {
        for (var b: u32 = a + 1u; b < 6u; b = b + 1u) {
            if (x0[a] > x0[b]) { rank[a] += 1u; } else { rank[b] += 1u; }
        }
    }

    var value : f32 = 0.0;
    for (var m: u32 = 0u; m <= 6u; m = m + 1u) {
        var off : array<i32, 6>;
        var xk  : array<f32, 6>;
        var r2  : f32 = 0.0;
        for (var a: u32 = 0u; a < 6u; a = a + 1u) {
            let o = select(0i, 1i, rank[a] < m);
            off[a] = o;
            let xa = x0[a] - f32(o) + f32(m) * G6;
            xk[a] = xa;
            r2 += xa * xa;
        }
        let t0 = ATTN6 - r2;
        if (t0 > 0.0) {
            let h = hash6(
                i[0] + off[0], i[1] + off[1], i[2] + off[2],
                i[3] + off[3], i[4] + off[4], i[5] + off[5]
            );
            let g = grad6(h);
            let dA = vec3<f32>(xk[0], xk[1], xk[2]);
            let dB = vec3<f32>(xk[3], xk[4], xk[5]);
            let k  = (t0 * t0);
            let w  = k * k;
            let s  = dot(g.ga, dA) + dot(g.gb, dB);
            value += w * s;
        }
    }
    return value * 50.0;
}

fn simplex6_fbm01(pos: vec3<f32>, seed: vec3<f32>) -> f32 {
    var f    : f32 = 0.0;
    var amp  : f32 = 1.0;
    var freq : f32 = 1.0;
    for (var o: u32 = 0u; o < noise6.octaves; o = o + 1u) {
        f    += amp * snoise6(pos, seed, freq);
        freq *= noise6.lacunarity;
        amp  *= noise6.gain;
    }
    return f;
}

fn coord_noise(coord: vec3<f32>, view_pos: vec3<f32>) -> vec3<f32> {
    let dis = length(view_pos - coord);
    let err = dis * dis;
    let err_dir = normalize(coord - view_pos);
    return err_dir * (simplex6_fbm01(coord, view_pos) * 0.4 * dis);
}

fn round_to_coord(v: vec3<f32>) -> vec3<i32> {
    let rv: vec3<f32> = round(v);
    return vec3<i32>(rv);
}

@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) wid: vec3<u32>) {
    let idx = wid.x;
    let visible = cull_params.visible_count;
    if (idx >= visible) { return; }

    let inst = culled_instances[idx];
    var coord = inst.position;
    if (noise6.enabled != 0u) {
        let camera_pos = camera.pos;
        let jitter = coord_noise(vec3<f32>(coord), camera_pos);
        coord = round_to_coord(vec3<f32>(coord) + jitter);
    }

    let out_idx = atomicAdd(&out_buf.count, 1u);
    out_buf.data[out_idx].coord = coord;
    out_buf.data[out_idx].value = inst.value;
}
