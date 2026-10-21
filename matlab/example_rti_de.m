% EXAMPLE_RTI_DE  Usage example of rti_de.m (MATLAB R2016b or later, or GNU Octave).
%
%   Run from the matlab/ folder:   example_rti_de
%   or from the command line:      octave-cli --eval "example_rti_de"

% Objective functions are vectorised: one point per row, one value per point.
rastrigin = @(X) 10 * size(X, 2) + sum(X .^ 2 - 10 * cos(2 * pi * X), 2);

D = 10;
lb = -5.12 * ones(1, D);
ub = 5.12 * ones(1, D);
max_fes = 50000;
seed = 1;

% Default settings: N_init = 18 D, N_min = 4, p = 0.11, CR = 0.9, chain after 70 % of the budget.
[x_best, f_best, fes] = rti_de(rastrigin, lb, ub, max_fes, seed);
fprintf('Rastrigin, D = %d: f_best = %.6g after %d evaluations\n', D, f_best, fes);

% Options are name-value pairs (or a struct); the fourth output is a per-generation trace.
[~, f2, ~, trace] = rti_de(rastrigin, lb, ub, max_fes, seed, 'n_init', 10 * D, 'chain_from', 0.5);
fprintf('n_init = 10 D, chain_from = 0.5: f_best = %.6g, %d generations, final mu_F = %.3f\n', ...
        f2, numel(trace), trace(end).mu_F);
ev = vertcat(trace.evaluated);
fprintf('candidates evaluated: %d first trial points, %d half steps, %d parabola points\n', sum(ev, 1));
