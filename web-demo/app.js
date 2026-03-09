/**
 * MotorSafe demo — quote form → pricing API → display result.
 * Set window.QUOTE_API_BASE and window.GITHUB_REPO_URL in index.html.
 */

(function () {
  const form = document.getElementById('quote-form');
  const resultEl = document.getElementById('quote-result');
  const errorEl = document.getElementById('quote-error');
  const errorMessageEl = document.getElementById('error-message');
  const submitBtn = document.getElementById('submit-btn');

  let lastQuotePayload = null;

  function getApiBase() {
    if (typeof window.QUOTE_API_BASE === 'string' && window.QUOTE_API_BASE) {
      return window.QUOTE_API_BASE.replace(/\/$/, '');
    }
    return '';
  }

  function getGitHubUrl() {
    return (typeof window.GITHUB_REPO_URL === 'string' && window.GITHUB_REPO_URL) ? window.GITHUB_REPO_URL : '#';
  }

  function wireGitHubLinks() {
    const url = getGitHubUrl();
    ['link-github-repo', 'nav-github', 'about-github', 'footer-github'].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) {
        el.href = url;
        if (id === 'nav-github' || id === 'about-github') el.target = '_blank';
      }
    });
    var docsEl = document.getElementById('footer-docs');
    if (docsEl && typeof window.FOOTER_DOCS_URL === 'string') docsEl.href = window.FOOTER_DOCS_URL;
    var portfolioEl = document.getElementById('footer-portfolio');
    if (portfolioEl && typeof window.FOOTER_PORTFOLIO_URL === 'string') portfolioEl.href = window.FOOTER_PORTFOLIO_URL;
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

  function formatPound(n) {
    if (n == null || !Number.isFinite(n)) return '—';
    return '£' + Number(n).toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function exposureLabel(exposure) {
    if (exposure == null) return '—';
    const e = Number(exposure);
    if (e <= 1/12) return '1 month';
    if (e <= 0.25) return '3 months';
    if (e <= 0.5) return '6 months';
    if (e <= 1) return '1 year';
    return e + ' years';
  }

  function humanizeTerm(term) {
    if (!term || typeof term !== 'string') return term;
    if (term.indexOf('VehBrand') !== -1) return 'Vehicle brand';
    if (term.indexOf('VehGas') !== -1) return 'Fuel type';
    if (term.indexOf('Area') !== -1) return 'Area';
    if (term.indexOf('Region') !== -1) return 'Region';
    if (term.indexOf('DrivAge') !== -1) return 'Driver age';
    if (term.indexOf('VehAge') !== -1) return 'Vehicle age';
    if (term.indexOf('BonusMalus') !== -1) return 'Bonus-Malus';
    if (term.indexOf('Density') !== -1 || term.indexOf('log1p') !== -1) return 'Population density';
    if (term.indexOf('VehPower') !== -1) return 'Vehicle power';
    if (term.indexOf('Exposure') !== -1) return 'Coverage period';
    return term;
  }

  function showResult(quotePayload) {
    lastQuotePayload = quotePayload;
    errorEl.hidden = true;
    resultEl.hidden = false;

    var payload = quotePayload;
    if (typeof quotePayload.body === 'string') {
      try {
        payload = JSON.parse(quotePayload.body);
      } catch (e) {
        payload = quotePayload;
      }
    }
    const quote = payload.quote || payload;
    const gross = quote.gross || {};
    const policy = quote.policy || {};
    const breakdown = gross.breakdown || {};
    const configVersion = quote.config_version || gross.pricing_config_version || '—';

    const amountEl = document.getElementById('quote-amount');
    const decisionEl = document.getElementById('quote-decision');
    const decisionBadgeEl = document.getElementById('quote-decision-badge');
    const configVersionEl = document.getElementById('quote-config-version');
    const riskEl = document.getElementById('quote-risk');
    const modelEl = document.getElementById('quote-model');
    const pricingEl = document.getElementById('quote-pricing');
    const summaryCardEl = document.getElementById('quote-summary-card');
    const breakdownCardEl = document.getElementById('quote-breakdown-card');
    const whyContentEl = document.getElementById('quote-why-content');

    if (gross && typeof gross.gross_premium === 'number') {
      amountEl.textContent = formatPound(gross.gross_premium);
    } else {
      amountEl.textContent = '—';
    }

    const decision = quote.decision || 'BIND';
    if (decisionBadgeEl) {
      decisionBadgeEl.textContent = decision;
      decisionBadgeEl.className = 'quote-decision-badge ' + (decision === 'REFER' ? 'refer' : 'bind');
    }
    if (configVersionEl) configVersionEl.textContent = 'Model version: ' + configVersion;

    decisionEl.textContent =
      decision === 'REFER'
        ? 'This quote requires manual review (refer to underwriter).'
        : 'Your quote is ready. This is a demo — no contract is formed.';

    // Quote summary card
    summaryCardEl.innerHTML =
      '<h4>Quote summary</h4><div class="quote-card-rows">' +
      '<div class="quote-card-row"><span>Quote decision</span><span>' + decision + '</span></div>' +
      '<div class="quote-card-row"><span>Policy duration</span><span>' + exposureLabel(policy.Exposure) + '</span></div>' +
      '<div class="quote-card-row"><span>Vehicle</span><span>' + codeToLabel('VehBrand', policy.VehBrand) + '</span></div>' +
      '<div class="quote-card-row"><span>Region</span><span>' + codeToLabel('Region', policy.Region) + '</span></div>' +
      '</div>';

    // Pricing breakdown card (from gross.breakdown)
    const pureVal = breakdown.pure_premium;
    const expenseVal = breakdown.expense_loading;
    const marginVal = breakdown.margin_loading;
    const grossAfter = breakdown.gross_after_caps != null ? breakdown.gross_after_caps : gross.gross_premium;
    breakdownCardEl.innerHTML =
      '<h4>Pricing breakdown</h4><div class="quote-card-rows">' +
      '<div class="quote-card-row"><span>Expected loss (pure premium)</span><span>' + formatPound(pureVal) + '</span></div>' +
      '<div class="quote-card-row"><span>Expense loading</span><span>' + formatPound(expenseVal) + '</span></div>' +
      '<div class="quote-card-row"><span>Margin loading</span><span>' + formatPound(marginVal) + '</span></div>' +
      '<div class="quote-card-row"><span>Final premium</span><span>' + formatPound(grossAfter) + ' / year</span></div>' +
      '</div>';

    // Risk assessment
    riskEl.innerHTML =
      '<h4>Risk assessment</h4><ul class="quote-summary">' +
      '<li>Driver age: ' + (policy.DrivAge != null ? policy.DrivAge : '—') + '</li>' +
      '<li>Vehicle: ' + codeToLabel('VehBrand', policy.VehBrand) + '</li>' +
      '<li>Location: ' + codeToLabel('Region', policy.Region) + ' (' + codeToLabel('Area', policy.Area) + ')</li>' +
      '</ul>';

    // Model outputs
    const pure = quote.pure || {};
    const freq = pure.lambda_freq != null ? Number(pure.lambda_freq).toFixed(2) : '—';
    const sev = pure.sev_mean != null ? formatPound(pure.sev_mean) : '—';
    modelEl.innerHTML =
      '<h4>Model outputs</h4><ul class="quote-summary">' +
      '<li>Expected claim frequency: ' + freq + ' claims/year</li>' +
      '<li>Expected claim severity: ' + sev + '</li>' +
      '</ul>';

    // Pricing block (short)
    const purePremium = pure.expected_loss != null ? formatPound(pure.expected_loss) : '—';
    const finalPremium = (gross && typeof gross.gross_premium === 'number') ? formatPound(gross.gross_premium) : '—';
    pricingEl.innerHTML =
      '<h4>Pricing</h4><ul class="quote-summary">' +
      '<li>Pure premium: ' + purePremium + '</li>' +
      '<li>Final premium: ' + finalPremium + ' / year</li>' +
      '</ul>';

    // Why this quote? (from explanation.top_features)
    const explanation = quote.explanation || {};
    const pureTop = (explanation.pure_premium && explanation.pure_premium.top_features) || [];
    const freqTop = (explanation.frequency && explanation.frequency.top_features) || [];
    const sevTop = (explanation.severity && explanation.severity.top_features) || [];
    const allTop = pureTop.length ? pureTop : (freqTop.length ? freqTop : sevTop);
    let whyHtml = '';
    if (allTop.length) {
      whyHtml = '<ul>';
      allTop.slice(0, 5).forEach(function (f) {
        const label = humanizeTerm(f.term);
        const mult = f.multiplicative_effect != null ? Number(f.multiplicative_effect).toFixed(2) : '—';
        const effect = mult !== '—' ? (mult > 1 ? mult + '× (increased risk)' : mult + '× (reduced risk)') : '';
        whyHtml += '<li><strong>' + label + '</strong> ' + effect + '</li>';
      });
      whyHtml += '</ul>';
    } else {
      whyHtml = '<p>Explanation data not available for this quote.</p>';
    }
    whyContentEl.innerHTML = whyHtml;
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
        showError(Array.isArray(data.detail) ? data.detail.map(function (x) { return x.msg || JSON.stringify(x); }).join(' ') : msg);
      }
    } catch (err) {
      showError(err.message || 'Network error. Is the API running?');
    } finally {
      submitBtn.disabled = false;
    }
  });

  var modal = document.getElementById('technical-details-modal');
  var modalPre = document.getElementById('technical-details-json');
  var btnTechnical = document.getElementById('btn-technical-details');
  var btnModalClose = document.getElementById('modal-close');

  if (btnTechnical && modal && modalPre) {
    btnTechnical.addEventListener('click', function () {
      if (lastQuotePayload) {
        modalPre.textContent = JSON.stringify(lastQuotePayload, null, 2);
        modal.showModal();
      } else {
        modalPre.textContent = 'No quote data yet. Get a quote first.';
        modal.showModal();
      }
    });
  }
  if (btnModalClose && modal) {
    btnModalClose.addEventListener('click', function () { modal.close(); });
  }
  if (modal) {
    modal.addEventListener('click', function (e) {
      if (e.target === modal) modal.close();
    });
  }

  wireGitHubLinks();
})();
