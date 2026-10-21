function run_octave_checks()
%RUN_OCTAVE_CHECKS  Octave/MATLAB side of the verification of rti_de.m (see ../VERIFICATION.md).
%
%   Run from this folder after "python verify_rtide.py prepare":
%       octave-cli --no-gui --quiet --eval "run_octave_checks"
%   Writes output/rules_oct, output/oct_<case>, output/counts_oct, output/stat_oct and
%   output/env_oct.json, which "python verify_rtide.py compare" reads.
global CNT_LOG
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                       % matlab/ (rti_de.m, rtide_rules.m)
out = fullfile(here, 'output');
R = rtide_rules();
cfg = jsondecode(fileread(fullfile(here, 'cases.json')));
t_start = tic;

% ------------------------------------------------------------------ known-answer checks
kat = {};
[t, ok] = R.parabola_factor([0; 0; 0; 0; 0; 0; 0; 1; 1], [10; 3; 2.1; 1; 2; 4; 5; Inf; 3], ...
                            [1; 1; 1; 1; 1; 1; 1; 2; Inf]);
kat{end + 1} = {'parabola: contraction t* = 6/32', t(1) == 0.1875 && ok(1)};
kat{end + 1} = {'parabola: extrapolation t* = -1/4', t(2) == -0.25 && ok(2)};
kat{end + 1} = {'parabola: t* = -4.75 clipped to -1', t(3) == -1 && ok(3)};
kat{end + 1} = {'parabola: not convex -> -1/2', t(4) == -0.5 && ok(4)};
kat{end + 1} = {'parabola: a = 0 -> -1/2', t(5) == -0.5 && ok(5)};
kat{end + 1} = {'parabola: t* = 0 is skipped', t(6) == 0 && ~ok(6)};
kat{end + 1} = {'parabola: t* = 1/12 kept', abs(t(7) - 1 / 12) < 1e-15 && ok(7)};
kat{end + 1} = {'parabola: infinite f(x + d) skipped', ~ok(8)};
kat{end + 1} = {'parabola: infinite half step skipped', ~ok(9)};
V = R.repair([-1, 3, 0.5], [0.4, 0.4, 0.4], [0, 0, 0], [1, 1, 1]);
kat{end + 1} = {'repair: midpoints 0.2 / 0.7, inside unchanged', isequal(V, [0.2, 0.7, 0.5])};
mu = R.update_muF(0.5, [2; 4], [1; 1], [0.2; 0.8], [1; 1]);
kat{end + 1} = {'mu_F: weighted Lehmer mean (gains 1, 3)', abs(mu - (0.45 + 0.1 * 0.49 / 0.65)) < 1e-15};
mu = R.update_muF(0.5, [Inf; Inf], [1; 2], [0.2; 0.8], [1; 1]);
kat{end + 1} = {'mu_F: infinite gains -> equal weights', abs(mu - 0.518) < 1e-15};
mu = R.update_muF(0.04, 2, 1, 0.05, 1);
kat{end + 1} = {'mu_F: clipped to 0.05', mu == 0.05};
kat{end + 1} = {'round half even', isequal(R.round_half_even([0.5 1.5 2.5 -0.5 -1.5 2.4 2.6]), [0 2 2 0 -2 2 3])};
kat{end + 1} = {'LPSR: 178.5 -> 178, 179.5 -> 180, end -> n_min', ...
    R.lpsr_size(4, 180, 352, 3) == 178 && R.lpsr_size(4, 180, 352, 1) == 180 && R.lpsr_size(4, 180, 1000, 1000) == 4};
kat{end + 1} = {'m = max(2, round(p N)): 2.5 -> 2, 4.5 -> 4, 19.8 -> 20, 0.44 -> 2', ...
    R.elite_count(0.125, 20) == 2 && R.elite_count(0.125, 36) == 4 && R.elite_count(0.11, 180) == 20 && R.elite_count(0.11, 4) == 2};
kat{end + 1} = {'npsum of 1..10 = 55', R.npsum(1:10) == 55};

% run-level checks (Octave random numbers)
lb = -5 * ones(1, 8); ub = 5 * ones(1, 8);
for mf = [1000, 2357]
    CNT_LOG = {};
    [x, fx, used] = rti_de(@counted_rastrigin, lb, ub, mf, 0);
    allX = vertcat(CNT_LOG{:});
    vals = rastrigin(allX);
    kat{end + 1} = {sprintf('budget %d: fes = evaluations = max_fes', mf), used == mf && size(allX, 1) == mf}; %#ok<AGROW>
    kat{end + 1} = {sprintf('budget %d: all points inside the box', mf), all(allX(:) >= -5) && all(allX(:) <= 5)}; %#ok<AGROW>
    kat{end + 1} = {sprintf('budget %d: returned value = f(x) = best evaluated', mf), ...
        fx == rastrigin(x) && fx == min(vals)}; %#ok<AGROW>
end
CNT_LOG = {};
rti_de(@counted_rastrigin, lb, ub, 3000, 0, 'n_init', 40, 'lpsr', false);
sizes = cellfun(@(A) size(A, 1), CNT_LOG(1:end - 1));
kat{end + 1} = {'no LPSR: every generation has n_init agents', all(sizes == 40)};
[~, ~, ~, tr] = rti_de(@rastrigin, -5 * ones(1, 10), 5 * ones(1, 10), 6000, 1);
ev = vertcat(tr.evaluated); fe = [tr.fes]; th = vertcat(tr.theta);
kat{end + 1} = {'trace: one candidate per evaluation', sum(ev(:)) == 6000 - 180 && fe(end) == 6000};
kat{end + 1} = {'trace: no chain before 0.7 max_fes', sum(sum(ev(fe < 0.7 * 6000, 2:3))) == 0 && sum(ev(:, 2)) > 0 && sum(ev(:, 3)) > 0};
kat{end + 1} = {'trace: t* in [-1, 1/4) and |t*| >= 0.02', all(th >= -1 & th < 0.25 & abs(th) >= 0.02)};
kat{end + 1} = {'trace: best value non-increasing', all(diff([tr.f_best]) <= 0)};
nexp = arrayfun(@(g) R.lpsr_size(4, 180, 6000, g.fes), tr(1:end - 1));
kat{end + 1} = {'trace: population size follows the LPSR schedule', isequal([tr(2:end).n], max(4, min(180, nexp)))};
[x1, f1] = rti_de(@rastrigin, -5 * ones(1, 6), 5 * ones(1, 6), 3000, 3);
[x2, f2] = rti_de(@rastrigin, -5 * ones(1, 6), 5 * ones(1, 6), 3000, 3);
kat{end + 1} = {'same seed, same run', isequal(x1, x2) && f1 == f2};
[~, fs] = rti_de(@(X) 500 + sum((X - 1.5) .^ 2, 2), -5 * ones(1, 10), 5 * ones(1, 10), 20000, 0);
kat{end + 1} = {'converges on a shifted sphere (error < 1e-6)', fs - 500 < 1e-6};

passed = sum(cellfun(@(c) logical(c{2}), kat));
for i = 1:numel(kat)
    fprintf('  [%s] %s\n', ifelse(kat{i}{2}, 'PASS', 'FAIL'), kat{i}{1});
end
fprintf('known-answer checks: %d of %d passed\n', passed, numel(kat));

% ------------------------------------------------------------------ rules on the Python inputs
I = read_arrays(fullfile(out, 'rules_in'));
O = struct();
[O.par_t, O.par_ok] = R.parabola_factor(I.par_fit, I.par_f_one, I.par_f_half);
O.rep_V = R.repair(I.rep_V, I.rep_X, I.rep_lb, I.rep_ub);
O.muF = zeros(I.ng, 1);
for g = 0:I.ng - 1
    p = sprintf('g%d_', g);
    O.muF(g + 1) = R.update_muF(I.([p 'mu']), I.([p 'fit']), I.([p 'fnew']), I.([p 'F']), I.([p 'T']));
end
O.lpsr = arrayfun(@(i) R.lpsr_size(I.lpsr(i, 1), I.lpsr(i, 2), I.lpsr(i, 3), I.lpsr(i, 4)), (1:size(I.lpsr, 1))');
O.elite = arrayfun(@(i) R.elite_count(I.elite(i, 1), I.elite(i, 2)), (1:size(I.elite, 1))');
O.sum_out = arrayfun(@(i) R.npsum(I.sum_in(i, 1:i)), (1:size(I.sum_in, 1))');
O.kat_passed = passed;
O.kat_total = numel(kat);
write_arrays(fullfile(out, 'rules_oct'), O);

% ------------------------------------------------------------------ trace-level replay
cases = cfg.trace_cases;
if ~iscell(cases)
    cases = num2cell(cases);
end
for i = 1:numel(cases)
    c = cases{i};
    f = objective(c.func);
    replay_source('load', fullfile(out, ['draws_' c.name]));
    opts = c.options;
    opts.rng_source = replay_source('source');
    [x, fx, used, tr] = rti_de(f, c.lb * ones(1, c.dim), c.ub * ones(1, c.dim), c.max_fes, [], opts);
    T = struct('x', x, 'f', fx, 'fes', used, 'gen_fes', [tr.fes]', 'gen_n', [tr.n]', ...
               'gen_mu_F', [tr.mu_F]', 'gen_f_best', [tr.f_best]', 'gen_evaluated', vertcat(tr.evaluated), ...
               'gen_accepted', vertcat(tr.accepted), 'gen_improved', vertcat(tr.improved), ...
               'gen_skipped', [tr.skipped]', 'theta', vertcat(tr.theta), ...
               'theta_accepted', double(vertcat(tr.theta_accepted)), 'draws_left', replay_source('remaining'));
    write_arrays(fullfile(out, ['oct_' c.name]), T);
    fprintf('  replay %s: %d generations, f = %.17g\n', c.name, numel(tr), fx);
end

% ------------------------------------------------------------------ evaluation counts
cc = cfg.count_cases;
if ~iscell(cc)
    cc = num2cell(cc);
end
rows = zeros(numel(cc), 4);
for i = 1:numel(cc)
    c = cc{i};
    CNT_LOG = {};
    [~, ~, used] = rti_de(@counted_rastrigin, c.lb * ones(1, c.dim), c.ub * ones(1, c.dim), c.max_fes, 0);
    rows(i, :) = [c.max_fes, used, sum(cellfun(@(A) size(A, 1), CNT_LOG)), numel(CNT_LOG)];
end
write_arrays(fullfile(out, 'counts_oct'), struct('rows', rows));

% ------------------------------------------------------------------ independent runs
st = cfg.stat;
fstar = struct('sphere', 0, 'rastrigin', 0, 'rosenbrock', 0);
t_stat = tic;
err = zeros(st.runs, numel(st.funcs));
for j = 1:numel(st.funcs)
    name = st.funcs{j};
    for r = 1:st.runs
        [~, fx] = rti_de(objective(name), st.lb * ones(1, st.dim), st.ub * ones(1, st.dim), st.max_fes, ...
                         st.seed_base + r - 1);
        err(r, j) = fx - fstar.(name);
    end
end
stat_runtime = toc(t_stat);
write_arrays(fullfile(out, 'stat_oct'), struct('err', err));

fid = fopen(fullfile(out, 'env_oct.json'), 'w');
fprintf(fid, '{\n  "octave": "%s",\n  "computer": "%s",\n  "rng": "rng(seed, ''twister'')",\n  "stat_runtime_s": %.1f,\n  "total_runtime_s": %.1f\n}\n', ...
        version, computer, stat_runtime, toc(t_start));
fclose(fid);
fprintf('octave checks done\n');
end


function f = objective(name)
switch name
    case 'sphere'
        f = @sphere;
    case 'shifted_sphere'
        f = @shifted_sphere;
    case 'rastrigin'
        f = @rastrigin;
    case 'rastrigin_nan'
        f = @rastrigin_nan;
    case 'rosenbrock'
        f = @rosenbrock;
    otherwise
        error('unknown function %s', name);
end
end

% Test functions: the row sums use the summation order of np.sum (rtide_rules().npsum_rows).
function v = sphere(X)
R = rtide_rules();
v = R.npsum_rows(X .* X);
end

function v = shifted_sphere(X)
R = rtide_rules();
Y = X - 1.5;
v = 500.0 + R.npsum_rows(Y .* Y);
end

function v = rastrigin(X)
R = rtide_rules();
v = 10 * size(X, 2) + R.npsum_rows(X .* X - 10 * cos(2 * pi * X));
end

function v = rastrigin_nan(X)
v = rastrigin(X);
v(X(:, 1) > 4.0) = NaN;
end

function v = rosenbrock(X)
R = rtide_rules();
A = X(:, 1:end - 1);
B = X(:, 2:end) - A .* A;
C = 1.0 - A;
v = R.npsum_rows(100.0 * (B .* B) + C .* C);
end

function v = counted_rastrigin(X)
global CNT_LOG
CNT_LOG{end + 1} = X;
v = rastrigin(X);
end

function s = ifelse(c, a, b)
if c
    s = a;
else
    s = b;
end
end
