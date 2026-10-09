// The examples: model texts in the language of expr.js. Each one is written operation for
// operation like its Julia twin in test/batched_models.jl (used by validate/reference.jl), so
// the Float32 browser result can be checked against the Float64 CPU solver (?validate=1).
//
// Fields: key, title, formula (HTML, shown under the selector), text, axes [x, y] (parameter
// names), p (steps per period), S (Gauss–Legendre stages), m (Krylov dimension), brute
// (default grid), mdbm (initial grid — 'bf': the brute-force grid — and iterations), bf / md
// (methods on by default), states (names of the state components), forced (periodic orbit on by
// default), note.

export const EXAMPLES = [
  {
    key: 'mathieu',
    title: 'Delayed damped Mathieu equation',
    formula: 'ẍ(t) + a₁ẋ(t) + (δ + ε cos t) x(t) = b₀ x(t − 2π)',
    text: `# delayed damped Mathieu equation
#   ẍ(t) + a₁ẋ(t) + (δ + ε cos t) x(t) = b₀ x(t − 2π)
# first-order form ẋ = A(t) x + B(t) x(t − τ), state x = (x, ẋ)
δ = -1:10 @ 3
ε = 0:10 @ 2
b₀ = -1:1 @ -0.15
a₁ = 0:1 @ 0.1
f₀ = 0:2 @ 1                  # forcing amplitude
T = 2*pi
τ = 2*pi
A = [0, 1; -(δ + ε*cos(t)), -a₁]
B = [0, 0; b₀, 0]
f = [0; f₀*cos(t)]            # forcing (periodic-orbit option)`,
    axes: ['δ', 'ε'], p: 16, S: 3, m: 6, states: ['x'], forced: false,
    brute: [256, 128], mdbm: ['bf', 0, 1],
  },
  {
    key: 'milling',
    title: '2-DOF milling (down-milling, textbook tool)',
    formula: 'q̈ + 2ζΩq̇ + Ω²q = −κ H(t) (q(t) − q(t − τ)), &nbsp;q = (x, y), time in units of 1/ω₁',
    text: `# 2-DOF milling, non-dimensional time t̃ = ω₁·t, state (x, y, ẋ/ω₁, ẏ/ω₁):
#   q̈ + 2ζ Ω_i q̇ + Ω_i² q = −κ H(t̃) (q(t̃) − q(t̃ − τ))
# tool: 922 Hz, m = 0.03993 kg, ζ = 0.011, 2 teeth, Kt = 6e8 N/m², Kn/Kt = 1/3
n = 5000:25000 @ 15000        # spindle speed [rpm]
w = 0:5 @ 2                   # axial depth of cut [mm]
aD = 0.02:1 @ 0.05            # radial immersion a/D
ζ = 0.002:0.05 @ 0.011        # damping ratio
ry = 0.8:1.3 @ 1.05           # y/x natural frequency ratio
fz = 0:0.3 @ 0.1              # feed per tooth [mm]
z = 2
ω₁ = 2*pi*922
κ = w*1e-3*6e8/(0.03993*ω₁^2)  # w Kt / (m ω₁²)
kn = 1/3
Ωs = 2*pi*n/60/ω₁             # spindle angular speed per unit t̃
φen = acos(2*aD - 1)          # entry angle (down-milling), exit at π
φ₁ = mod(Ωs*t, 2*pi)
φ₂ = mod(Ωs*t + 2*pi/z, 2*pi)
g₁ = step(φ₁ - φen)*step(pi - φ₁)
g₂ = step(φ₂ - φen)*step(pi - φ₂)
hxx = g₁*(cos(φ₁) + kn*sin(φ₁))*sin(φ₁) + g₂*(cos(φ₂) + kn*sin(φ₂))*sin(φ₂)
hxy = g₁*(cos(φ₁) + kn*sin(φ₁))*cos(φ₁) + g₂*(cos(φ₂) + kn*sin(φ₂))*cos(φ₂)
hyx = g₁*(-sin(φ₁) + kn*cos(φ₁))*sin(φ₁) + g₂*(-sin(φ₂) + kn*cos(φ₂))*sin(φ₂)
hyy = g₁*(-sin(φ₁) + kn*cos(φ₁))*cos(φ₁) + g₂*(-sin(φ₂) + kn*cos(φ₂))*cos(φ₂)
T = ω₁*60/(z*n)               # tooth-passing period = delay
τ = ω₁*60/(z*n)
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     -1 - κ*hxx, -κ*hxy, -2*ζ, 0;
     -κ*hyx, -ry^2 - κ*hyy, 0, -2*ζ*ry]
B = [0, 0, 0, 0;
     0, 0, 0, 0;
     κ*hxx, κ*hxy, 0, 0;
     κ*hyx, κ*hyy, 0, 0]
# cutting force with the nominal chip thickness fz·sin φ (periodic-orbit option), x, y in mm
f = [0; 0; -κ*fz*hxx; -κ*fz*hyx]`,
    axes: ['n', 'w'], p: 40, S: 3, m: 8, states: ['x [mm]'], forced: true,
    brute: [256, 128], mdbm: ['bf', 0, 1],
    note: 'The cutting-force coefficient switches when a tooth enters or leaves the cut, so the ' +
          'step functions make A(t), B(t) discontinuous: expect first-order convergence in p. ' +
          'MDBM finds a lobe only if the initial grid resolves its width (narrow low-speed lobes need a finer initial grid).',
  },
  {
    key: 'turning_ssv',
    title: 'Turning with spindle-speed variation (time-periodic delay)',
    formula: 'ẍ + ζẋ + x = k<sub>w</sub> (x(t − τ(t)) − x(t)), &nbsp;τ(t) = 2π/Ω (1 + RVA sin(RVF·Ω t)), &nbsp;T = 2π/(RVF·Ω)',
    text: `# turning with sinusoidal spindle-speed variation (non-dimensional)
#   ẍ + ζẋ + x = k_w (x(t − τ(t)) − x(t)),  τ(t) = 2π/Ω (1 + RVA sin(RVF·Ω t))
#   RVA: relative variation amplitude, RVF: relative variation frequency (period T = 2π/(RVF·Ω))
Ω = 0.2:2 @ 1                 # mean spindle speed
k_w = 0:0.6 @ 0.2             # cutting-force coefficient
RVA = 0:0.3 @ 0.1             # relative speed-variation amplitude
RVF = 0.05:0.5 @ 0.1          # relative speed-variation frequency
ζ = 0.01:0.3 @ 0.1            # damping
h₀ = 0:1 @ 0.1                # nominal chip thickness (feed per revolution)
T = 2*pi/(RVF*Ω)
τ = 2*pi/Ω*(1 + RVA*sin(RVF*Ω*t))
A = [0, 1; -1 - k_w, -ζ]
B = [0, 0; k_w, 0]
# cutting force with the nominal chip thickness, which follows the varying delay (periodic-orbit option)
f = [0; k_w*h₀*(1 + RVA*sin(RVF*Ω*t))]`,
    axes: ['Ω', 'k_w'], p: 200, S: 3, m: 6, states: ['x'], forced: false,
    brute: [256, 128], mdbm: ['bf', 0, 1],
    note: 'One period T holds 1/RVF delays, so one delay gets p·RVF steps (20 at the defaults): for a small RVF raise p.',
  },
];
