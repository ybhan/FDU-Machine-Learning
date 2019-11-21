function [kernel_c, kernel_f, weight_f, weight_output, bias_c,bias_f] = ...
    CNN_update(stepSize, classify, train_data, state_c, state_f, ...
    state_f_temp, output, kernel_c, kernel_f, weight_f, weight_output, ...
    bias_c, bias_f)
% CNN_UPDATE computes the gradients of weights, kernels and bias, and then
% updates these parameters by stepSize using gradient descent method.
%
% Yuanbo Han, Dec. 9, 2017.
% Reference: http://blog.csdn.net/u010540396/article/details/52895074

% Compute the number of neurons and some sizes of matrices.
nHidden = size(state_f,2);
nLabels = size(output,2);
[c_row, c_col, nConv] = size(state_c);
[kernel_c_row, kernel_c_col] = size(kernel_c(:,:,1));
[kernel_f_row, kernel_f_col] = size(kernel_f(:,:,1));

% The temp values will record the updated values of parameters, in order
% not to overwrite the original values.
kernel_c_temp = kernel_c;
kernel_f_temp = kernel_f;
weight_f_temp = weight_f;
weight_output_temp = weight_output;

% Compute error.
label = zeros(1, nLabels);
label(1, classify) = 1;
delta_layer_output = output - label;

% Update weight_output.
delta_weight_output = zeros(nHidden, nLabels);
for n = 1:nLabels
    delta_weight_output(:,n) = delta_layer_output(1,n) * state_f';
end
weight_output_temp = weight_output_temp - stepSize * delta_weight_output;

% Update full-connected-layer parameters (kernel_f, bias_f, weights_f).
delta_bias_f = zeros(1, nHidden);
delta_kernel_f = zeros(kernel_f_row, kernel_f_col, nHidden);
delta_layer_f = zeros(1, nHidden);
for n = 1:nHidden
    count = 0;
    for m = 1:nLabels
        count = count + delta_layer_output(1,m) * weight_output(n,m);
    end
    % update bias_f
    delta_layer_f(1,n) = count * (1 - tanh(state_f(1,n)).^2);
    delta_bias_f(1,n) = delta_layer_f(1,n);
    % update kernel_f
    delta_kernel_f(:,:,n) = delta_layer_f(1,n) * state_f_temp(:,:,n);
end
bias_f = bias_f - stepSize * delta_bias_f;
kernel_f_temp = kernel_f_temp - stepSize * delta_kernel_f;

% update weight_f
delta_layer_f_temp = zeros(kernel_f_row, kernel_f_col, nHidden);
for n = 1:nHidden
    delta_layer_f_temp(:,:,n) = delta_layer_f(1,n) * kernel_f(:,:,n);
end
delta_weight_f = zeros(nConv, nHidden);
for n = 1:nConv
    for m = 1:nHidden
        delta_weight_f(n,m) = sum(sum( delta_layer_f_temp(:,:,m) .* ...
            state_c(:,:,n) ));
    end
end
weight_f_temp = weight_f_temp - stepSize * delta_weight_f;

% Update convolution-layer parameters (i.e. kernel_c, bias_c).
% update bias_c
delta_layer_c = zeros(c_row, c_col, nConv);
delta_bias_c = zeros(1, nConv);
for n = 1:nConv
    count = 0;
    for m = 1:nHidden
        count = count + delta_layer_f_temp(:,:,m) * weight_f(n,m);
    end
    delta_layer_c(:,:,n) = count .* ( sech(state_c(:,:,n)).^2 );
    delta_bias_c(1,n) = sum(sum(delta_layer_c(:,:,n)));
end
bias_c = bias_c - stepSize * delta_bias_c;

% update kernel_c
delta_kernel_c_temp = zeros(kernel_c_row, kernel_c_col, nConv);
for n = 1:nConv
    delta_kernel_c_temp(:,:,n) = rot90( conv2(train_data, ...
        rot90(delta_layer_c(:,:,n),2), 'valid'), 2);
end
kernel_c_temp = kernel_c_temp - stepSize * delta_kernel_c_temp;

% Final overwriting
kernel_c = kernel_c_temp;
kernel_f = kernel_f_temp;
weight_f = weight_f_temp;
weight_output = weight_output_temp;

end
