function [S,T,w,b] = train_csp(EEG,Fs,mrk,wnd,f,nof,n)
% Train a CSP+LDA classifier
% [S,T,w,b] = train_csp(RawSignal, SampleRate, Markers, EpochWnd, SpectralFlt, FltNumber, FltLength)
%
% In:
%   RawSignal : raw data array [#samples x #channels]
%
%   SampleRate : sampling rate of the data, in Hz
%
%   Markers : marker channel (0 = no marker, 1 = first class, 2 = second class)
%
%   EpochWnd : time range of the epochs to extract relative to the marker
%              in seconds ([begin, end]), e.g. [0.5 3.5]
%
%   SpectralFlt : spectral filter specfication; this is a function of Frequency in Hz
%                 (e.g., f = @(x)x>7&x<30)
%
%   FltNumber : number of spatial filters pairs to compute (e.g., 3)
%
%   FltLength : length of the temporal filter, in samples (e.g., 200)
%
% Out:
%   S : spatial filter matrix [#channels x #filters]
%
%   T : temporal filter matrix [FltLength x FltLength]
%
%   w : linear classifier weights
%
%   b : linear classifier bias


% do frequency filtering using FFT
[t,c] = size(EEG); idx = reshape(1:t*c-mod(t*c,n),n,[]);
FLT = real(ifft(fft(EEG).*repmat(f(Fs*(0:t-1)/t)',1,c)));

% estimate temporal filter using least-squares
T = FLT(idx)/EEG(idx);

% extract data for all epochs of the first class concatenated (EPO{1}) and 
% all epochs of the second class concatenated (EPO{2})
% each array is [#samples x #channels]
wnd = round(Fs*wnd(1)) : round(Fs*wnd(2));
for k = 1:2
    EPO{k} = FLT(repmat(find(mrk==k),length(wnd),1) + repmat(wnd',1,nnz(mrk==k)),:);
end

% calculate the spatial filter matrix S using CSP (TODO: fill in)
E1 = cov(EPO{1});
E2 = cov(EPO{2});

[S,D]= eig(E1,E1 + E2);

% log-variance feature extraction
for k = 1:2
    X{k} = squeeze(log(var(reshape(EPO{k}*S, length(wnd),[],2*nof))));
end

S = S(:,1:6);
filtered1 = EPO{1} * S;
filtered2 = EPO{2} * S;

% train LDA classifier (preferably with gradual outputs) (TODO: fill in)
E1 = cov(filtered1);
E2 = cov(filtered2);
u1 = transpose(mean(filtered1));
u2 = transpose(mean(filtered2));
w = inv(E1 + E2)*(u2-u1);
b = -0.5 * ( transpose(w) * (u1 + u2));

