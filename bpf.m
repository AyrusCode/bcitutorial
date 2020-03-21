function signal = bpf(Fs, oldsignal)
    dt = 1/Fs;
    signal = oldsignal';
    N = size(signal, 1);
    dF = Fs/N;
    f = (-Fs/2:dF:Fs/2-dF)';
    % Band-Pass Filter:
    BPF = ((0.1 < abs(f)) & (abs(f) < 40));

    % time=0.008:0.008:size(signal,1)*0.008;
    time = dt*(0:N-1)';


   % spektrum = fft(signal,NFFT)/(size(signal,1));
    spektrum = fftshift(fft(signal))/N;

    % spektrum(lower_freq:upper_freq,1)=0;
    % spektrum(size(spektrum,1)- upper_freq+1:size(spektrum,1)-lower_freq+1,1)=0;
    spektrum = BPF.*spektrum;
    
    % signal=ifft(spektrum); %inverse ifft
    signal=ifft(ifftshift(spektrum)); %inverse ifft

end
