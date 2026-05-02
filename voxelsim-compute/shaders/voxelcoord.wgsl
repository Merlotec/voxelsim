const FILLED: u32 = 16;
const SPARSE: u32 = 32;
const GROUND: u32 = 64;
const TARGET: u32 = 128;
const INTEREST: u32 = 256;

// The camera data structure, matching the Rust struct.
struct CameraUniform {
    view_proj: mat4x4<f32>,
    pos: vec3<f32>,
}

// Use a push constant for the camera data. This removes the need for a bind group.
var<push_constant> camera: CameraUniform;

struct InstanceInput {
    @location(1) coord: vec3<i32>,
    @location(2) value: u32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) coord: vec3<i32>,
    @location(1) value: u32,
}

// =================== Parameters (uniform) =====================
struct Noise6Params {
  spatial_freq : vec3<f32>, // cycles / world unit for x,y,z (e.g., vec3(0.5))
  seed_freq    : vec3<f32>, // cycles / unit for sx,sy,sz    (e.g., vec3(0.2))
  seed_vec     : vec3<f32>, // your continuous seed vector
  lacunarity   : f32,       // e.g., 2.0
  gain         : f32,       // e.g., 0.5
  octaves      : u32,       // e.g., 4
  enabled       : u32,       // alignment padding
}
@group(0) @binding(0) var<uniform> noise6 : Noise6Params;

// =================== Constants & helpers ======================
// 6D skew/unskew constants (Gustavson N-D simplex)
const F6 : f32 = (sqrt(7.0) - 1.0) / 6.0;
const G6 : f32 = (1.0 - 1.0 / sqrt(7.0)) / 6.0;

// Attenuation kernel: t = ATTN6 - |x|^2, contribution ~ t^4 * (g·x).
const ATTN6 : f32 = 0.5;

fn mix32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u; h *= 0x7feb352du;
    h ^= h >> 15u; h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

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

fn randSym(seed: u32) -> f32 {
    let r = mix32(seed);
    return f32(r) * (1.0 / 4294967295.0) * 2.0 - 1.0; // [-1,1]
}

struct G6s { ga: vec3<f32>, gb: vec3<f32> }  // 6D gradient = (ga | gb)

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

// =================== 6D simplex core (f32 out) ==============
fn snoise6(pos: vec3<f32>, seed: vec3<f32>, freq_scale: f32) -> f32 {
    // Pack the 6D point: (x,y,z, sx,sy,sz) with per-axis frequencies
    var p : array<f32, 6>;
    p[0] = pos.x  * (noise6.spatial_freq.x * freq_scale);
    p[1] = pos.y  * (noise6.spatial_freq.y * freq_scale);
    p[2] = pos.z  * (noise6.spatial_freq.z * freq_scale);
    p[3] = seed.x * (noise6.seed_freq.x    * freq_scale);
    p[4] = seed.y * (noise6.seed_freq.y    * freq_scale);
    p[5] = seed.z * (noise6.seed_freq.z    * freq_scale);

    // Skew into simplex grid
    var sumP : f32 = 0.0;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { sumP += p[a]; }
    let s : f32 = sumP * F6;

    // Base lattice coordinates
    var i : array<i32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { i[a] = i32(floor(p[a] + s)); }

    // Unskew
    var sumI : i32 = 0;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { sumI += i[a]; }
    let t : f32 = f32(sumI) * G6;

    var x0 : array<f32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { x0[a] = p[a] - f32(i[a]) + t; }

    // Rank-based corner selection: rank[a] = how many components are greater than x0[a]
    var rank : array<u32, 6>;
    for (var a: u32 = 0u; a < 6u; a = a + 1u) { rank[a] = 0u; }
    for (var a: u32 = 0u; a < 6u; a = a + 1u) {
        for (var b: u32 = a + 1u; b < 6u; b = b + 1u) {
            if (x0[a] > x0[b]) { rank[a] += 1u; } else { rank[b] += 1u; }
        }
    }

    var value : f32 = 0.0;

    // 7 simplex corners: m = 0..6
    for (var m: u32 = 0u; m <= 6u; m = m + 1u) {
        var off : array<i32, 6>;
        var xk  : array<f32, 6>;
        var r2  : f32 = 0.0;

        // For corner m, add 1 to the m largest components (rank < m)
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
            let w  = k * k; // t^4
            let s  = dot(g.ga, dA) + dot(g.gb, dB);

            value += w * s;
        }
    }

    // Empirical scale for a comfortable range; tweak 40–55 to taste.
    return value * 50.0;
}

// =================== fBm wrappers ==============================
fn simplex6_fbm(pos: vec3<f32>, seed: vec3<f32>) -> f32 {
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

// Remap to roughly [0,1]
fn simplex6_fbm01(pos: vec3<f32>, seed: vec3<f32>) -> f32 {
    return simplex6_fbm(pos, seed);
}
// This is our noise function to create the randomness.
// We require a few things:
// The uncertainty must be continuous - i.e. if we move a small amount we should expect a small change in our sample error of a specific area.
// Hence, the sampling distribution should be parameterised by the camera position.
// So we parameterise a noise function basaed on the camera position. The seed must have a continuous effect on the noise function.
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

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    let camera_pos = camera.pos;
    if noise6.enabled != 0 {    
        let instance_pos = vec3<f32>(instance.coord);
        let noise = coord_noise(instance_pos, camera_pos);
        out.coord = round_to_coord(instance_pos + noise);
    } else {
        out.coord = instance.coord;
    }
    
    let world_position = vec4<f32>(model.position + vec3<f32>(out.coord), 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.value = instance.value;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<i32> {
    return vec4<i32>(in.coord, bitcast<i32>(in.value));
}
