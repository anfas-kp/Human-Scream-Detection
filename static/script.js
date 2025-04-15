document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    let audioContext, source, processor;

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const status = document.getElementById('status');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const maxAmplitude = document.getElementById('max_amplitude');
    const meanAmplitude = document.getElementById('mean_amplitude');
    const stdDev = document.getElementById('std_dev');
    const waveformPlot = document.getElementById('waveformPlot');
    const histogramPlot = document.getElementById('histogramPlot');
    const plotsDiv = document.getElementById('realtimePlots');
    const themeToggle = document.getElementById('themeToggle');

    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');
    let audioBuffer = [];

    // Debug background image
    const bodyStyle = window.getComputedStyle(document.body);
    const backgroundImage = bodyStyle.backgroundImage;
    console.log('Body background image:', backgroundImage);
    if (backgroundImage === 'none') {
        console.error('Background image not applied! Check CSS or file path.');
    }

    // Theme toggle
    themeToggle.addEventListener('click', () => {
        document.body.dataset.theme = document.body.dataset.theme === 'dark' ? '' : 'dark';
    });

    socket.on('status', (data) => {
        status.textContent = data.message;
    });

    socket.on('prediction', (data) => {
        prediction.textContent = data.label;
        prediction.className = data.label === 'Scream' ? 'scream' : 'not-scream';
        confidence.textContent = data.confidence;
        maxAmplitude.textContent = data.features.max_amplitude.toFixed(4);
        meanAmplitude.textContent = data.features.mean_amplitude.toFixed(4);
        stdDev.textContent = data.features.std_dev.toFixed(4);

        if (data.plots) {
            const timestamp = new Date().getTime();
            const waveformUrl = `/static/${data.plots.waveform}?${timestamp}`;
            const histogramUrl = `/static/${data.plots.histogram}?${timestamp}`;
            
            setTimeout(() => {
                waveformPlot.src = waveformUrl;
                histogramPlot.src = histogramUrl;
                plotsDiv.style.display = 'block';
                console.log('Real-time plots updated:', waveformUrl, histogramUrl);

                waveformPlot.onerror = () => console.error('Failed to load waveform:', waveformUrl);
                histogramPlot.onerror = () => console.error('Failed to load histogram:', histogramUrl);
            }, 100);
        } else {
            plotsDiv.style.display = 'none';
        }
    });

    function drawWaveform(data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.strokeStyle = '#007BFF';
        ctx.lineWidth = 1;

        const sliceWidth = canvas.width / data.length;
        let x = 0;

        for (let i = 0; i < data.length; i++) {
            const y = (data[i] + 1) * canvas.height / 2;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }
        ctx.stroke();
    }

    startBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext({ sampleRate: 16000 });
            source = audioContext.createMediaStreamSource(stream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                const audioData = e.inputBuffer.getChannelData(0);
                audioBuffer = audioBuffer.concat(Array.from(audioData));
                if (audioBuffer.length > 800) {
                    audioBuffer = audioBuffer.slice(-800);
                }
                drawWaveform(audioBuffer);
                socket.emit('audio_data', audioData.buffer);
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.textContent = 'Listening...';
        } catch (err) {
            status.textContent = 'Error accessing microphone: ' + err.message;
        }
    });

    stopBtn.addEventListener('click', () => {
        if (processor) processor.disconnect();
        if (source) source.disconnect();
        if (audioContext) audioContext.close();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        status.textContent = 'Stopped';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        plotsDiv.style.display = 'none';
        audioBuffer = [];
    });
});