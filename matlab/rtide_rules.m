function R = rtide_rules()
%RTIDE_RULES  Update rules of RTI-DE as separate functions (used by rti_de.m and by the tests).
%
%   R = rtide_rules() returns a struct of function handles:
%
%     R.round_half_even(x)            rounding to the nearest integer, ties to the even integer
%                                     (Python's round(), used for m and for the population size)
%     R.npsum(v)                      sum of a vector in the order of NumPy's pairwise summation
%     R.npsum_rows(A)                 the same for every row of a matrix (np.sum(A, axis=1))
%     R.elite_count(p, n)             m = max(2, round(p n)), number of best agents for x_e
%     R.repair(V, X, lb, ub)          midpoint repair of the mutant points V (rows) of the agents X
%     R.parabola_factor(fit, f_one, f_half)
%                                     [t, ok]: factor t* of the third point and ok = false when the
%                                     skip rule applies (or an increase is not finite)
%     R.update_muF(mu_F, fit, f_new, F, T)
%                                     new mu_F from the accepted first trial points (vectors over
%                                     the agents counted: f_new < fit and stage 0)
%     R.lpsr_size(n_min, n_init, max_fes, fes)
%                                     population size after a generation (before max with n)
%
%   Every rule follows python/RTIDE/optimizer.py line by line, including the order of the floating
%   point operations, so that a run with the same random numbers gives the same values.

R.round_half_even = @round_half_even;
R.npsum = @npsum;
R.npsum_rows = @npsum_rows;
R.elite_count = @elite_count;
R.repair = @repair;
R.parabola_factor = @parabola_factor;
R.update_muF = @update_muF;
R.lpsr_size = @lpsr_size;
end


function y = round_half_even(x)
% Python round(): nearest integer, a tie goes to the even integer.
y = round(x);
tie = abs(x - fix(x)) == 0.5;
y(tie) = 2 * round(x(tie) / 2);
end


function s = npsum(v)
% Sum of the elements of v in the order used by NumPy (pairwise summation, see npsum_rows).
s = npsum_rows(reshape(v, 1, []));
end


function s = npsum_rows(A)
% Row sums in the order of NumPy's pairwise summation of float64 (numpy/_core/src/umath/
% loops_utils.h.src, pairwise_sum): fewer than 8 values are added in sequence; up to 128 values
% are added into 8 partial sums that are combined as ((r1+r2)+(r3+r4))+((r5+r6)+(r7+r8)), the
% remainder being added in sequence; longer rows are split in two parts (the first of length
% floor(n/2) rounded down to a multiple of 8) that are summed recursively.
[k, n] = size(A);
if n < 8
    s = zeros(k, 1);
    for j = 1:n
        s = s + A(:, j);
    end
elseif n <= 128
    r = A(:, 1:8);
    i = 9;
    lim = n - mod(n, 8);
    while i <= lim
        r = r + A(:, i:i + 7);
        i = i + 8;
    end
    s = ((r(:, 1) + r(:, 2)) + (r(:, 3) + r(:, 4))) + ((r(:, 5) + r(:, 6)) + (r(:, 7) + r(:, 8)));
    for j = i:n
        s = s + A(:, j);
    end
else
    n2 = floor(n / 2);
    n2 = n2 - mod(n2, 8);
    s = npsum_rows(A(:, 1:n2)) + npsum_rows(A(:, n2 + 1:end));
end
end


function m = elite_count(p, n)
% Number of best agents from which x_e is drawn: max(2, round(p n)).
m = max(2, round_half_even(p * n));
end


function V = repair(V, X, lb, ub)
% A component outside the box is set to the midpoint between the violated bound and the agent
% (applied to the mutant point, before the crossover). Rows of V and X are agents.
lo = V < lb;
M = (lb + X) / 2.0;
V(lo) = M(lo);
hi = V > ub;
M = (ub + X) / 2.0;
V(hi) = M(hi);
end


function [t, ok] = parabola_factor(fit, f_one, f_half)
% Factor t* of the third point from the values at 0, 1/2 and 1 (column vectors over the agents).
d_one = f_one - fit;
d_half = f_half - fit;
beta = (d_half - 0.5 * d_one) / -0.25;          % coefficient a of theta^2
t = -(d_one - beta) ./ (2.0 * beta);            % minimiser -b / (2 a), b = d_one - a
t(~(isfinite(t) & (beta > 0))) = -0.5;          % not convex (or a = 0): fixed backward step
ok = isfinite(d_one) & isfinite(d_half);
t(~ok) = 0.0;
t = min(max(t, -1.0), 1.0);
ok = ok & (abs(t) >= 0.02) & (abs(t - 0.5) >= 0.02) & (abs(t - 1.0) >= 0.02);   % skip rule
end


function mu_F = update_muF(mu_F, fit, f_new, F, T)
% mu_F <- clip(0.9 mu_F + 0.1 L, 0.05, 1), L the Lehmer mean of the step factors weighted by
% the improvement (inputs: the agents counted, in the order of their index).
gain = fit - f_new;
gain(~isfinite(gain)) = 0.0;
g = npsum(gain);
if g > 0
    w = gain / npsum(gain);
else
    w = repmat(1.0 / numel(gain), size(gain));
end
s_f = F .* abs(T);
den = npsum(w .* s_f);
if den > 0
    lehmer = npsum(w .* (s_f .* s_f)) / den;
else
    lehmer = 0.0;
end
mu_F = min(max(0.9 * mu_F + 0.1 * lehmer, 0.05), 1.0);
end


function n_next = lpsr_size(n_min, n_init, max_fes, fes)
% Linear population size reduction: max(n_min, round((n_min - n_init) / max_fes * fes + n_init)).
n_next = max(n_min, round_half_even((n_min - n_init) / max_fes * fes + n_init));
end
