function [state_f,state_f_temp] = convolution_f(state_c,kernel_f,weight_f)
% CONVOLUTION_f computes the full-connected-layer values.
%
% Yuanbo Han, Dec. 9, 2017.

[nConv, nHidden] = size(weight_f);
[c_row, c_col, ~] = size(state_c);
f_row = size(state_c,1) - size(kernel_f,1) + 1;
f_col = size(state_c,2) - size(kernel_f,2) + 1;

state_f = zeros(f_row, f_col, nHidden);
state_f_temp = zeros(c_row, c_col, nHidden);
for n = 1:nHidden
    count = 0;
    for m = 1:nConv
        count = count + state_c(:,:,m) * weight_f(m,n);
    end
    state_f_temp(:,:,n) = count;
    state_f(:,:,n) = conv2(state_f_temp(:,:,n), ...
        rot90(kernel_f(:,:,n),2), 'valid');
end

end
