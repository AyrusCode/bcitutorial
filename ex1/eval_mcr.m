function loss = eval_mcr(predictions,targets)
% Evaluate the mis-classification rate loss
% Loss = eval_mcr(Predictions,Targets)
%
% In:
%   Predictions : vector of predictions made by the classifier
%
%   Targets : vector of true labels (-1 / +1)
%
% Out:
%   Loss : mis-classification rate

% calculate mis-classification rate (TODO: fill in) 
% True label - prediction label
sum = 0;
n = length(predictions)
for i = 1:n
    if predictions(i) == targets(i)
        sum = sum + 1;
    end
end

loss = sum/n
