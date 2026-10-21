function [x_best, f_best, fes, trace] = rti_de(func, lb, ub, max_fes, seed, varargin)
%RTI_DE  Rejected-Trial Interpolation Differential Evolution (MATLAB / GNU Octave port).
%
%   [x_best, f_best, fes] = rti_de(func, lb, ub, max_fes)
%   [x_best, f_best, fes] = rti_de(func, lb, ub, max_fes, seed)
%   [x_best, f_best, fes, trace] = rti_de(func, lb, ub, max_fes, seed, 'Name', value, ...)
%
%   Minimises func within max_fes function evaluations over the box [lb, ub]. func is vectorised:
%   it receives a k-by-D matrix (one point per row) and returns k objective values. A value NaN is
%   treated as +Inf.
%
%   This is a port of python/RTIDE/optimizer.py (function rti_de); the rules, their order, the
%   order of the random draws and the evaluation accounting are the same. The random numbers are
%   those of MATLAB/Octave, so a run with a given seed differs from the Python run with that seed
%   (see matlab/VERIFICATION.md for the equivalence check with injected random numbers).
%
%   seed        [] (default): the global random number generator is used as it is; otherwise
%               rng(seed, 'twister') is called first (this changes the global generator state).
%
%   Options (name-value pairs, or one struct with these fields); defaults as in Python:
%     'n_init'      initial population size ([] or 0: 18 D)
%     'n_min'       final population size (4)
%     'p'           fraction of the population from which x_e is drawn (0.11)
%     'chain'       points of the rejected-trial chain: 2 (half step and parabola point),
%                   1 (half step only) or 0 (no chain) (2)
%     'chain_from'  fraction of the budget after which the chain is used (0.7)
%     'CR'          rate of the binomial crossover; [] means no crossover (0.9)
%     'lpsr'        false: the population keeps its initial size (true)
%     'adapt'       false: mu_F stays at 0.5 (true)
%     'rng_source'  struct of function handles that supply the random numbers (for tests):
%                     uniform(n, D)  n-by-D uniform numbers in [0, 1)  (initial population)
%                     randint(h, k)  k-by-1 integers, uniform on 1..h
%                     randn(k)       k-by-1 standard normal numbers
%                     rand(k, D)     k-by-D uniform numbers in [0, 1)  (crossover)
%                   Default: rand, randi and randn of MATLAB/Octave.
%
%   Outputs
%     x_best   best point (1-by-D), f_best its value, fes the number of evaluations used
%     trace    (only when requested) struct array, one element per generation, with the fields of
%              the Python trace: fes, n, mu_F, f_best, evaluated, accepted, improved (1-by-3, by
%              stage [first trial point, half step, parabola point]), skipped, theta,
%              theta_accepted. Building it draws no random number and evaluates nothing.
%
%   Algorithm (see the paper, Section 3, and Algorithm 1):
%     1. Trial point v = x + F (x_e - x) + F (x_r1 - x_r2) (current-to-pbest/1 without archive),
%        x_e drawn from the best max(2, round(p N)) agents; components outside the box are set
%        to the midpoint between the bound and x; binomial crossover with rate CR and at least
%        one component from v. The displacement is d = trial point - x.
%     2. Selection: the candidate replaces x when f(candidate) <= f(x).
%     3. Rejected-trial chain once FE >= chain_from * max_fes: after a rejected trial point the
%        agent evaluates x + d/2; after a rejected half step it evaluates x + t* d, t* the
%        minimiser of the parabola through (0, f(x)), (1/2, f(x + d/2)), (1, f(x + d)), limited
%        to [-1, 1], -1/2 if the parabola is not convex; skipped (a new trial point is drawn
%        instead) when t* is closer than 0.02 to 0, 1/2 or 1. The candidate is clipped to the box.
%     4. F ~ N(mu_F, 0.15) clipped to [0.05, 1]; mu_F <- clip(0.9 mu_F + 0.1 L, 0.05, 1) with L
%        the Lehmer mean of the F of the improving first trial points, weighted by improvement.
%     5. Linear population size reduction from n_init to n_min (worst agents removed).
%
%   Example
%     f = @(X) sum(X.^2, 2);
%     [x, fx, used] = rti_de(f, -5 * ones(1, 10), 5 * ones(1, 10), 20000, 1);

if nargin < 5
    seed = [];
end
opt = parse_options(varargin);
R = rtide_rules();

lb = double(reshape(lb, 1, []));
ub = double(reshape(ub, 1, []));
dim = numel(lb);
if numel(ub) ~= dim
    error('rti_de:bounds', 'lb and ub must have the same number of elements');
end
if isempty(opt.n_init) || opt.n_init == 0
    n_init = 18 * dim;
else
    n_init = fix(double(opt.n_init));
end
n_min = opt.n_min;
p = opt.p;
chain = opt.chain;
chain_from = opt.chain_from;
CR = opt.CR;
src = opt.rng_source;
if isempty(src)
    if ~isempty(seed)
        rng(seed, 'twister');
    end
    src = struct('uniform', @(n, d) rand(n, d), 'randint', @(h, k) randi(h, k, 1), ...
                 'randn', @(k) randn(k, 1), 'rand', @(k, d) rand(k, d));
end
want_trace = nargout >= 4;
trace = struct('fes', {}, 'n', {}, 'mu_F', {}, 'f_best', {}, 'evaluated', {}, 'accepted', {}, ...
               'improved', {}, 'skipped', {}, 'theta', {}, 'theta_accepted', {});

n = n_init;
X = lb + (ub - lb) .* src.uniform(n, dim);
fit = evaluate(func, X);
fes = n;
mu_F = 0.5;
stage = zeros(n, 1);          % 0: first trial point, 1: half step, 2: parabola point
Dm = zeros(n, dim);           % displacement of the running chain
T = ones(n, 1);               % factor of d in the current candidate
F = repmat(0.5, n, 1);        % step factor with which d was drawn
f_one = zeros(n, 1);          % value at x + d
f_half = zeros(n, 1);         % value at x + d / 2

while fes < max_fes
    % --- third point: minimiser of the parabola through (0, fit), (1/2, f_half), (1, f_one)
    s2 = find(stage == 2);
    n_skip = 0;
    if ~isempty(s2)
        [t, ok] = R.parabola_factor(fit(s2), f_one(s2), f_half(s2));
        T(s2(ok)) = t(ok);
        stage(s2(~ok)) = 0;
        n_skip = sum(~ok);
    end

    % --- second point: half step
    T(stage == 1) = 0.5;

    % --- first trial point: new displacement
    s0 = find(stage == 0);
    if ~isempty(s0)
        k = numel(s0);
        [~, order] = sort(fit);                          % stable, as argsort(kind="stable")
        elite = X(order(src.randint(R.elite_count(p, n), k)), :);
        f = min(max(mu_F + 0.15 * src.randn(k), 0.05), 1.0);
        r1 = src.randint(n - 1, k);
        r1 = r1 + (r1 >= s0);
        r2 = src.randint(n, k);
        while true
            bad = (r2 == s0) | (r2 == r1);
            if ~any(bad)
                break;
            end
            r2(bad) = src.randint(n, sum(bad));
        end
        Xs = X(s0, :);
        V = Xs + f .* (elite - Xs) + f .* (X(r1, :) - X(r2, :));
        V = R.repair(V, Xs, lb, ub);
        if ~isempty(CR)
            cross = src.rand(k, dim) < CR;
            jr = src.randint(dim, k);
            cross(sub2ind([k, dim], (1:k)', jr)) = true;
            V(~cross) = Xs(~cross);
        end
        Dm(s0, :) = V - Xs;
        F(s0) = f;
        T(s0) = 1.0;
    end

    U = min(max(X + T .* Dm, lb), ub);
    k = min(n, max_fes - fes);                           % the last generation is cut at the budget
    f_new = inf(n, 1);
    f_new(1:k) = evaluate(func, U(1:k, :));
    fes = fes + k;

    better = f_new < fit;
    accept = f_new <= fit;
    accept(k + 1:end) = false;

    % --- step factor: accepted first trial points only
    counted = better & (stage == 0);
    if opt.adapt && any(counted)
        mu_F = R.update_muF(mu_F, fit(counted), f_new(counted), F(counted), T(counted));
    end

    if want_trace
        done = (1:n)' <= k;
        ev = zeros(1, 3); ac = zeros(1, 3); im = zeros(1, 3);
        for s = 0:2
            m = done & (stage == s);
            ev(s + 1) = sum(m); ac(s + 1) = sum(m & accept); im(s + 1) = sum(m & better);
        end
        m2 = done & (stage == 2);
        trace(end + 1) = struct('fes', fes, 'n', n, 'mu_F', mu_F, ...
            'f_best', min(min(fit), min(f_new(1:k))), 'evaluated', ev, 'accepted', ac, ...
            'improved', im, 'skipped', n_skip, 'theta', T(m2), 'theta_accepted', accept(m2)); %#ok<AGROW>
    end

    % --- selection and next stage of the chain
    rejected = ~accept;
    rejected(k + 1:end) = false;
    m = rejected & (stage == 0);
    f_one(m) = f_new(m);
    m = rejected & (stage == 1);
    f_half(m) = f_new(m);
    X(accept, :) = U(accept, :);
    fit(accept) = f_new(accept);
    nxt = zeros(n, 1);
    nxt(rejected) = stage(rejected) + 1;
    nxt(nxt > chain) = 0;
    if fes < chain_from * max_fes
        nxt(:) = 0;
    end
    nxt(k + 1:end) = stage(k + 1:end);
    stage = nxt;

    % --- linear population size reduction
    if opt.lpsr
        n_next = R.lpsr_size(n_min, n_init, max_fes, fes);
    else
        n_next = n;
    end
    if n_next < n
        [~, order] = sort(fit);
        keep = order(1:n_next);
        X = X(keep, :); fit = fit(keep); stage = stage(keep); Dm = Dm(keep, :);
        T = T(keep); F = F(keep); f_one = f_one(keep); f_half = f_half(keep);
        n = n_next;
    end
end

[f_best, best] = min(fit);                               % first minimum, as numpy.argmin
x_best = X(best, :);
end


function f = evaluate(func, X)
f = double(func(X));
f = reshape(f, [], 1);
f(isnan(f)) = Inf;
end


function opt = parse_options(args)
opt = struct('n_init', [], 'n_min', 4, 'p', 0.11, 'chain', 2, 'chain_from', 0.7, 'CR', 0.9, ...
             'lpsr', true, 'adapt', true, 'rng_source', []);
if numel(args) == 1 && isstruct(args{1})
    s = args{1};
    names = fieldnames(s);
    args = cell(1, 2 * numel(names));
    for i = 1:numel(names)
        args{2 * i - 1} = names{i};
        args{2 * i} = s.(names{i});
    end
end
if mod(numel(args), 2) ~= 0
    error('rti_de:options', 'options must be name-value pairs');
end
valid = fieldnames(opt);
for i = 1:2:numel(args)
    name = args{i};
    j = find(strcmpi(valid, name), 1);
    if isempty(j)
        error('rti_de:options', 'unknown option ''%s''', name);
    end
    opt.(valid{j}) = args{i + 1};
end
end
