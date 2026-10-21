function out = replay_source(cmd, varargin)
%REPLAY_SOURCE  Supplies to rti_de.m the random numbers recorded from a run of the Python code.
%
%   replay_source('load', prefix)   load the draws <prefix>.idx/.bin written by verify_rtide.py
%   src = replay_source('source')   struct of handles for the 'rng_source' option of rti_de
%   n = replay_source('remaining')  number of recorded draws not yet used
%
%   Each recorded draw has a kind (1 uniform, 2 randint, 3 randn, 4 rand), the upper bound of a
%   randint draw and its shape. A request whose kind, bound or shape differs from the next
%   recorded draw stops with an error: the control flow of the two implementations differs.
%   randint values are recorded 0-based (NumPy) and returned 1-based.
persistent kind high shape vals pos
switch cmd
    case 'load'
        prefix = varargin{1};
        fid = fopen([prefix '.idx'], 'r');
        C = textscan(fid, '%f %f %f %f');
        fclose(fid);
        kind = C{1}; high = C{2}; shape = [C{3} C{4}];
        fid = fopen([prefix '.bin'], 'r', 'ieee-le');
        allv = fread(fid, Inf, 'double');
        fclose(fid);
        vals = cell(numel(kind), 1);
        o = 0;
        for i = 1:numel(kind)
            r = shape(i, 1); c = shape(i, 2);
            vals{i} = reshape(allv(o + 1:o + r * c), c, r)';
            o = o + r * c;
        end
        if o ~= numel(allv)
            error('replay_source:size', 'index and data do not match');
        end
        pos = 0;
        out = numel(kind);
    case 'source'
        out = struct('uniform', @(n, d) next_draw(1, 0, [n d]), ...
                     'randint', @(h, k) next_draw(2, h, [k 1]) + 1, ...
                     'randn', @(k) next_draw(3, 0, [k 1]), ...
                     'rand', @(k, d) next_draw(4, 0, [k d]));
    case 'remaining'
        out = numel(kind) - pos;
    case 'next'
        want = varargin{1}; h = varargin{2}; shp = varargin{3};
        pos = pos + 1;
        if pos > numel(kind)
            error('replay_source:exhausted', 'more random numbers requested than recorded');
        end
        if kind(pos) ~= want || high(pos) ~= h || any(shape(pos, :) ~= shp)
            error('replay_source:mismatch', ...
                  'draw %d: requested kind %d, bound %d, shape %dx%d; recorded kind %d, bound %d, shape %dx%d', ...
                  pos, want, h, shp(1), shp(2), kind(pos), high(pos), shape(pos, 1), shape(pos, 2));
        end
        out = vals{pos};
    otherwise
        error('replay_source:cmd', 'unknown command %s', cmd);
end
end


function v = next_draw(kind, h, shp)
v = replay_source('next', kind, h, shp);
end
