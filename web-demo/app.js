/**
 * MotorSafe demo — quote form → pricing API → display result.
 * Set window.QUOTE_API_BASE to your API base URL (e.g. http://localhost:8000 for FastAPI).
 */

(function () {
  const form = document.getElementById('quote-form');
  const resultEl = document.getElementById('quote-result');
  const errorEl = document.getElementById('quote-error');
  const errorMessageEl = document.getElementById('error-message');
  const submitBtn = document.getElementById('submit-btn');

  // API base URL: use window.QUOTE_API_BASE or default to same origin
  function getApiBase() {
    if (typeof window.QUOTE_API_BASE === 'string' && window.QUOTE_API_BASE) {
      return window.QUOTE_API_BASE.replace(/\/$/, '');
    }
    return '';
  }

  function buildPolicyFromForm() {
    const num = (id) => {
      const el = document.getElementById(id);
      const v = el && el.value ? parseFloat(el.value, 10) : NaN;
      return Number.isFinite(v) ? v : null;
    };
    const str = (id) => {
      const el = document.getElementById(id);
      return el && el.value ? String(el.value).trim() : null;
    };

    return {
      Area: str('Area') || 'A',
      VehPower: num('VehPower') ?? 6,
      VehAge: num('VehAge') ?? 0,
      DrivAge: num('DrivAge') ?? 35,
      BonusMalus: num('BonusMalus') ?? 80,
      VehBrand: str('VehBrand') || 'B1',
      VehGas: (str('VehGas') || 'regular').toLowerCase(),
      Density: num('Density') ?? 1000,
      Region: str('Region') || 'R24',
      Exposure: num('Exposure') ?? 1,
    };
  }

  function codeToLabel(field, code) {
    const map = window.MOTORSAFE_LABELS && window.MOTORSAFE_LABELS[field];
    return (map && map[code]) ? map[code] : code;
  }

  function formatEur(n) {
    if (n == null || !Number.isFinite(n)) return '—';
    return '£' + Number(n).toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function showResult(quotePayload) {
    errorEl.hidden = true;
    resultEl.hidden = false;

    const quote = quotePayload.quote || quotePayload;
    const gross = quote.gross;
    const policy = quote.policy || {};
    const amountEl = document.getElementById('quote-amount');
    const decisionEl = document.getElementById('quote-decision');
    const riskEl = document.getElementById('quote-risk');
    const modelEl = document.getElementById('quote-model');
    const pricingEl = document.getElementById('quote-pricing');

    if (gross && typeof gross.gross_premium === 'number') {
      amountEl.textContent = formatEur(gross.gross_premium);
    } else {
      amountEl.textContent = '—';
    }

    const decision = quote.decision || 'BIND';
    decisionEl.textContent =
      decision === 'REFER'
        ? 'This quote requires manual review (refer to underwriter).'
        : 'Your quote is ready. This is a demo — no contract is formed.';

    // Risk assessment: driver, vehicle, location (human labels)
    riskEl.innerHTML =
      '<h4>Risk assessment</h4><ul class="quote-summary">' +
      '<li>Driver age: ' + (policy.DrivAge != null ? policy.DrivAge : '—') + '</li>' +
      '<li>Vehicle: ' + codeToLabel('VehBrand', policy.VehBrand) + '</li>' +
      '<li>Location: ' + codeToLabel('Region', policy.Region) + ' (' + codeToLabel('Area', policy.Area) + ')</li>' +
      '</ul>';

    // Model outputs: frequency, severity (friendly wording)
    const pure = quote.pure || {};
    const freq = pure.lambda_freq != null ? Number(pure.lambda_freq).toFixed(2) : '—';
    const sev = pure.sev_mean != null ? formatEur(pure.sev_mean) : '—';
    modelEl.innerHTML =
      '<h4>Model outputs</h4><ul class="quote-summary">' +
      '<li>Expected claim frequency: ' + freq + ' claims/year</li>' +
      '<li>Expected claim severity: ' + sev + '</li>' +
      '</ul>';

    // Pricing: pure premium, final premium
    const purePremium = pure.expected_loss != null ? formatEur(pure.expected_loss) : '—';
    const finalPremium = (gross && typeof gross.gross_premium === 'number') ? formatEur(gross.gross_premium) : '—';
    pricingEl.innerHTML =
      '<h4>Pricing</h4><ul class="quote-summary">' +
      '<li>Pure premium: ' + purePremium + '</li>' +
      '<li>Final premium: ' + finalPremium + ' / year</li>' +
      '</ul>';
  }

  function showError(message) {
    resultEl.hidden = true;
    errorEl.hidden = false;
    errorMessageEl.textContent = message || 'Unknown error.';
  }

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const base = getApiBase();
    const url = base ? base + '/quote' : '/quote';

    submitBtn.disabled = true;
    resultEl.hidden = true;
    errorEl.hidden = true;

    const policy = buildPolicyFromForm();
    const body = JSON.stringify(policy);

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body,
      });
      const data = await res.json().catch(() => ({}));

      if (res.ok) {
        showResult(data);
        document.getElementById('quote-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        const msg = data.detail || data.message || (typeof data.detail === 'string' ? data.detail : 'Request failed.');
        showError(Array.isArray(data.detail) ? data.detail.map((x) => x.msg || JSON.stringify(x)).join(' ') : msg);
      }
    } catch (err) {
      showError(err.message || 'Network error. Is the API running?');
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
