function S = read_arrays(prefix)
%READ_ARRAYS  Read arrays written by write_arrays (Python or Octave): <prefix>.idx and <prefix>.bin.
%   The index has one line per array, "name rows cols"; the values are float64, little endian,
%   row-major, concatenated in the order of the index.
fid = fopen([prefix '.idx'], 'r');
C = textscan(fid, '%s %f %f');
fclose(fid);
fid = fopen([prefix '.bin'], 'r', 'ieee-le');
S = struct();
for i = 1:numel(C{1})
    r = C{2}(i); c = C{3}(i);
    v = fread(fid, r * c, 'double');
    S.(C{1}{i}) = reshape(v, c, r)';
end
fclose(fid);
end
