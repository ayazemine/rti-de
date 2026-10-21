function write_arrays(prefix, S)
%WRITE_ARRAYS  Write the fields of struct S (2-D numeric or logical arrays) as <prefix>.idx/.bin
%   (float64, little endian, row-major), the format read by read_arrays and verify_rtide.py.
names = fieldnames(S);
fi = fopen([prefix '.idx'], 'w');
fb = fopen([prefix '.bin'], 'w', 'ieee-le');
for i = 1:numel(names)
    A = double(S.(names{i}));
    if isempty(A)
        A = zeros(0, 0);
    end
    fprintf(fi, '%s %d %d\n', names{i}, size(A, 1), size(A, 2));
    fwrite(fb, A', 'double');
end
fclose(fi);
fclose(fb);
end
