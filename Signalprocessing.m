%fare qui il processing del segnale del sensore
    Fs = 100; % Sampling frequency
    [b2, a2] = butter(11, 10/(Fs/2)); % low-pass filter used for smoothing
    finalresultsmag = filtfilt(b2, a2, finalresultsmag);
    finalresultsphase = filtfilt(b2, a2, finalresultsphase);
    %filtro in dc also
    