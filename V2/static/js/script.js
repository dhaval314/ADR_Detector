document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.querySelector('.predict-btn');
    btn.classList.add('loading');

    const formData = {
        age: document.getElementById('age').value,
        weight: document.getElementById('weight').value,
        dosage: document.getElementById('dosage').value,
        reactions: document.getElementById('reactions').value,
        concomitant_drugs: document.getElementById('concomitant_drugs').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        document.getElementById('results').style.display = 'block';
        document.getElementById('riskLevelDisplay').textContent = `Risk Level: ${data.risk_level}`;
        const riskPercent = Math.min(data.prediction * 100, 100);
        document.getElementById('meterFill').style.width = `${riskPercent}%`;
        document.getElementById('riskLabel').textContent = `${data.risk_level} (${riskPercent.toFixed(1)}%)`;

        // Render SHAP heatmaps
        const shapContainer = document.getElementById('shapContainer');
        shapContainer.innerHTML = ''; // Clear previous graphs

        data.shap_plots.forEach((plot, index) => {
            const plotWrapper = document.createElement('div');
            plotWrapper.className = 'plot-wrapper';

            const title = document.createElement('h4');
            title.textContent = index === 0 ? 'Reaction Heatmap' : 'Feature Impact Heatmap';

            const imgElement = document.createElement('img');
            imgElement.src = `data:image/png;base64,${plot}`;
            imgElement.className = 'shap-plot heatmap';

            plotWrapper.appendChild(title);
            plotWrapper.appendChild(imgElement);
            shapContainer.appendChild(plotWrapper);
        });

    } catch (error) {
        alert(`Prediction failed: ${error.message}`);
    } finally {
        btn.classList.remove('loading');
    }
});