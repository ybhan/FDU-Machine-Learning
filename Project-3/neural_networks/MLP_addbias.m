function [f,g] = MLP_addbias(w,X,y,nHidden,nLabels)
% MLP_ADDBIAS makes one of the hidden units in each layer a constant, so
% that each layer has a bias.
%
% Yuanbo Han, Dec. 8, 2017.

[nInstances, nVars] = size(X);
nHiddenLayers = length(nHidden);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars * nHidden(1);
hiddenWeights = cell(1, nHiddenLayers-1);
for h = 2:nHiddenLayers
    hiddenWeights{h-1} = reshape(...
        w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),...
        nHidden(h-1)+1, nHidden(h));
    offset = offset + (nHidden(h-1) + 1) * nHidden(h);
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels);
outputWeights = reshape(outputWeights, nHidden(end)+1, nLabels);

ip = cell(1, nHiddenLayers);
fp = cell(1, nHiddenLayers);
if nargout > 1
    % Form Gradient
    gInput = zeros(size(inputWeights));
    gHidden = cell(1, nHiddenLayers-1);
    for h = 2:nHiddenLayers
        gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
    end
    gOutput = zeros(size(outputWeights));
    
    f = 0;
    
    % Compute Output
    for i = 1:nInstances
        ip{1} = [1, X(i,:)*inputWeights];
        fp{1} = tanh(ip{1});
        for h = 2:length(nHidden)
            ip{h} = [1, fp{h-1}*hiddenWeights{h-1}];
            fp{h} = tanh(ip{h});
        end
        yhat = fp{end} * outputWeights;
        
        relativeErr = yhat - y(i,:);
        f = f + sum(relativeErr.^2);
        
        err = 2 * relativeErr;
        
        % Output Weights
        gOutput = gOutput + fp{end}' * err;
        
        % to be modified
        if nHiddenLayers > 1
            % Last Layer of Hidden Weights
            backprop = err' * sech(ip{end}).^2 .* outputWeights';
            gHidden{end} = gHidden{end} + fp{end-1}' * sum(backprop,1);
            
            backprop = sum(backprop,1);
            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end
            
            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}).^2;
            gInput = gInput + X(i,:)' * backprop;
            
        else % nHiddenLayers == 1
            % Input Weights
            temp = sech(ip{end}).^2 .* (outputWeights * err')';
            gInput = gInput + X(i,:)' * temp(2:end);
        end
    end
    
    % Put Gradient into vector
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:nHiddenLayers
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1};
        offset = offset+(nHidden(h-1)+1)*nHidden(h);
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gOutput(:);
    
    
else % nargout <= 1
    ip{1} = [ones(ninstances,1), X*inputWeights];
    fp{1} = tanh(ip{1});
    for h = 2:nHiddenLayers
        ip{h} = [ones(ninstances,1), fp{h-1}*hiddenWeights{h-1}];
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end} * outputWeights;
    
    relativeErr = yhat - y;
    f = sum(sum(relativeErr.^2));
end

end
