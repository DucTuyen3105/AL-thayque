const form = document.querySelector('#form');
const input = document.querySelector('#name');
const submit = form.querySelector('.submit');
const submitLabel = document.querySelector('#submit-label');
let requestNumber = 0;
form.addEventListener('submit', async event => {
  event.preventDefault();
  const current = ++requestNumber;
  submit.disabled = true;
  submitLabel.textContent = 'Analyzing…';
  document.querySelector('#result').setAttribute('aria-busy', 'true');
  document.querySelector('#error').hidden = true;
  document.querySelector('#output').hidden = true;
  try {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: input.value, mode: 'auto'})
    });
    const data = await response.json();
    if (current !== requestNumber) return;
    if (!response.ok) throw new Error(data.error || 'Unable to analyze this name.');
    document.querySelector('#placeholder').hidden = true;
    document.querySelector('#output').hidden = false;
    document.querySelector('#status').textContent = data.status;
    document.querySelector('#origin').textContent = data.candidates[0].language;
    document.querySelector('#normalized').textContent = `Normalized name: ${data.normalized}`;
    const bars = document.querySelector('#bars');
    bars.replaceChildren();
    for (const candidate of data.candidates) {
      const row = document.createElement('div'); row.className = 'bar-row';
      const caption = document.createElement('div'); caption.className = 'bar-caption';
      const label = document.createElement('span'); label.textContent = candidate.language;
      const score = document.createElement('span'); score.textContent = `${(candidate.score * 100).toFixed(1)}%`;
      caption.append(label, score);
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${candidate.score * 100}%`;
      track.append(fill); row.append(caption, track); bars.append(row);
    }
    const warnings = document.querySelector('#warnings'); warnings.replaceChildren();
    if (data.uncertain && !data.warnings.length) data.warnings.push('This name does not clearly distinguish one origin from the others.');
    for (const warning of data.warnings) {
      const paragraph = document.createElement('p'); paragraph.className = 'warning'; paragraph.textContent = warning; warnings.append(paragraph);
    }
    document.querySelector('#components').textContent = data.components.length > 1
      ? data.components.map(p => `${p.name}: ${p.language} ${(p.score * 100).toFixed(1)}%`).join(' · ') : '';
  } catch (error) {
    if (current !== requestNumber) return;
    document.querySelector('#placeholder').hidden = true;
    const box = document.querySelector('#error');
    box.textContent = error instanceof TypeError ? 'Unable to connect. Please try again.' : error.message;
    box.hidden = false;
  } finally {
    if (current === requestNumber) {
      submit.disabled = false; submitLabel.textContent = 'Analyze name';
      document.querySelector('#result').setAttribute('aria-busy', 'false');
    }
  }
});
