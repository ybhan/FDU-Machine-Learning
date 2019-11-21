function [yhat] = CNN_predict(X, kernel_c, kernel_f, weight_f, ...
    weight_output, bias_c, bias_f)
% CNN_predict classifies X by CNN model.
%
% Yuanbo Han, Dec. 9, 2017.

nInstances = size(X, 3);
yhat = zeros(nInstances, 1);
nConv = size(kernel_c, 3);
[nHidden, nLabels] = size(weight_output);
c_row = size(X,1) - size(kernel_c,1) + 1;
c_col = size(X,2) - size(kernel_c,2) + 1;

for i = 1:nInstances
    train_data = X(:,:,i);
    
    % Convolution layer
    state_c = zeros(c_row, c_col, nConv);
    for k = 1:nConv
        state_c(:,:,k) = conv2(train_data, ...
            rot90(kernel_c(:,:,k),2),'valid');
        % apply tanh
        state_c(:,:,k) = tanh(state_c(:,:,k) + bias_c(1,k));
    end
    
    % Full-connected layer
    [state_f_pre,~] = convolution_f(state_c, kernel_f, weight_f);
    % apply tanh
    state_f = zeros(1, nHidden);
    for h = 1:nHidden
        state_f(1,h) = tanh(state_f_pre(:,:,h) + bias_f(1,h));
    end
    
    % Output layer (Softmax)
    output = zeros(1, nLabels);
    for h = 1:nLabels
        output(1,h) = exp( state_f * weight_output(:,h) ) / ...
            sum( exp(state_f * weight_output) );
    end
    [~, yhat(i)] = max(output);
end

end
