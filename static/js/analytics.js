/**
 * Analytics Dashboard — Amazon-Style Charts
 * Chart.js powered CTR analytics, A/B test results, and timeline charts.
 */

let ctrChart = null;
let timelineChart = null;

// Amazon-themed chart colors
const CHART_COLORS = {
    collaborative: { main: '#ff9900', bg: 'rgba(255,153,0,0.15)' },
    content_based: { main: '#007185', bg: 'rgba(0,113,133,0.15)' },
    hybrid:        { main: '#232f3e', bg: 'rgba(35,47,62,0.15)' },
    popular:       { main: '#b12704', bg: 'rgba(177,39,4,0.15)' },
};

const ALGO_LABELS = {
    collaborative: 'Collaborative Filtering',
    content_based: 'Content-Based',
    hybrid: 'AI Hybrid',
    popular: 'Popular (Baseline)',
};

async function loadAnalytics() {
    try {
        const ctrData = await api('/analytics/ctr');
        renderAnalyticsCards(ctrData);
        renderCTRChart(ctrData);
        const timeline = await api('/analytics/ctr-timeline?days=30');
        renderTimelineChart(timeline);
    } catch (e) { console.error('Failed to load analytics:', e); }
}

function renderAnalyticsCards(data) {
    const container = document.getElementById('analyticsCards');
    const cards = [
        { icon: '👁️', value: (data.total_impressions || 0).toLocaleString(), label: 'Total Impressions' },
        { icon: '🖱️', value: (data.total_clicks || 0).toLocaleString(), label: 'Total Clicks' },
        { icon: '📈', value: `${data.overall_ctr || 0}%`, label: 'Overall CTR' },
        { icon: '🧪', value: data.total_sessions || 0, label: 'Test Sessions' },
    ];
    container.innerHTML = cards.map(c => `
        <div class="stat-card">
            <div class="stat-icon">${c.icon}</div>
            <div class="stat-value">${c.value}</div>
            <div class="stat-label">${c.label}</div>
        </div>`).join('');
}

function renderCTRChart(data) {
    const ctx = document.getElementById('ctrChart');
    if (!ctx) return;
    if (ctrChart) ctrChart.destroy();

    const algorithms = Object.keys(data.ctr_by_algorithm || {});
    const ctrValues = algorithms.map(a => data.ctr_by_algorithm[a].ctr || 0);
    const clickValues = algorithms.map(a => data.ctr_by_algorithm[a].clicks || 0);
    const bgColors = algorithms.map(a => (CHART_COLORS[a] || { main: '#999' }).main);

    ctrChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: algorithms.map(a => ALGO_LABELS[a] || a),
            datasets: [{
                label: 'CTR (%)',
                data: ctrValues,
                backgroundColor: bgColors.map(c => c + 'cc'),
                borderColor: bgColors,
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.6,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#232f3e', titleColor: '#fff', bodyColor: '#ccc',
                    borderColor: '#ff9900', borderWidth: 1, cornerRadius: 6, padding: 10,
                    callbacks: { afterBody: ctx => `Clicks: ${clickValues[ctx[0].dataIndex]}` },
                },
            },
            scales: {
                y: { beginAtZero: true, grid: { color: '#e3e6e6' }, ticks: { color: '#565959', callback: v => v + '%' }, title: { display: true, text: 'Click-Through Rate (%)', color: '#565959' } },
                x: { grid: { display: false }, ticks: { color: '#565959' } },
            },
        },
    });
}

function renderTimelineChart(data) {
    const ctx = document.getElementById('ctrTimelineChart');
    if (!ctx) return;
    if (timelineChart) timelineChart.destroy();

    const datasets = Object.keys(data).map(algo => ({
        label: ALGO_LABELS[algo] || algo,
        data: (data[algo] || []).map(d => ({ x: d.date, y: d.ctr })),
        borderColor: (CHART_COLORS[algo] || { main: '#999' }).main,
        backgroundColor: (CHART_COLORS[algo] || { bg: 'rgba(0,0,0,0.05)' }).bg,
        borderWidth: 2, fill: true, tension: 0.4, pointRadius: 3, pointHoverRadius: 6,
    }));

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { color: '#565959', usePointStyle: true, pointStyle: 'circle', padding: 20 } },
                tooltip: { backgroundColor: '#232f3e', titleColor: '#fff', bodyColor: '#ccc', borderColor: '#ff9900', borderWidth: 1, cornerRadius: 6, padding: 10 },
            },
            scales: {
                y: { beginAtZero: true, grid: { color: '#e3e6e6' }, ticks: { color: '#565959', callback: v => v + '%' }, title: { display: true, text: 'CTR (%)', color: '#565959' } },
                x: { type: 'category', grid: { display: false }, ticks: { color: '#565959', maxRotation: 45 } },
            },
        },
    });
}

async function loadABTestResults() {
    const container = document.getElementById('abTestResultsContainer');
    try {
        const data = await api('/analytics/ab-test');
        if (data.error) {
            container.innerHTML = `<div class="empty-state"><div class="empty-icon">🧪</div><p>${data.error}</p></div>`;
            return;
        }

        let html = '';
        const sig = data.significance || {};
        const sigClass = sig.significant ? 'significant' : 'not-significant';
        const sigIcon = sig.significant ? '✅' : '⚠️';

        html += `
            <div class="glass-card" style="margin-bottom:16px;">
                <h3 style="margin-bottom:12px;">Statistical Significance</h3>
                <div class="significance-badge ${sigClass}">${sigIcon} ${sig.message || 'No data available'}</div>
                ${sig.p_value !== undefined ? `
                    <div style="margin-top:12px;color:#565959;font-size:0.85rem;">
                        <strong>p-value:</strong> ${sig.p_value} |
                        <strong>χ² statistic:</strong> ${sig.chi2_statistic} |
                        <strong>DoF:</strong> ${sig.degrees_of_freedom}
                    </div>` : ''}
            </div>`;

        html += `
            <div class="glass-card">
                <h3 style="margin-bottom:12px;">Algorithm Comparison</h3>
                <table class="results-table">
                    <thead><tr><th>Algorithm</th><th>Sessions</th><th>Impressions</th><th>Clicks</th><th>Avg CTR</th><th>Total CTR</th><th>Status</th></tr></thead>
                    <tbody>`;

        const icons = { collaborative: '🤝', content_based: '📝', hybrid: '🧠', popular: '📊' };
        (data.groups || []).forEach(group => {
            const isWinner = group.algorithm_group === data.winner;
            html += `<tr>
                <td><strong>${icons[group.algorithm_group] || ''} ${ALGO_LABELS[group.algorithm_group] || group.algorithm_group}</strong></td>
                <td>${group.num_sessions}</td>
                <td>${group.total_impressions.toLocaleString()}</td>
                <td>${group.total_clicks.toLocaleString()}</td>
                <td><strong>${group.avg_ctr}%</strong></td>
                <td>${group.total_ctr}%</td>
                <td>${isWinner ? '<span class="winner-badge">🏆 Winner</span>' : ''}</td>
            </tr>`;
        });

        html += '</tbody></table></div>';

        html += `<div class="chart-container" style="margin-top:16px;"><h3>A/B Test — Click Distribution</h3><div class="chart-wrapper"><canvas id="abTestChart"></canvas></div></div>`;

        container.innerHTML = html;
        renderABTestChart(data.groups || []);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Failed to load A/B test results.</p></div>`;
    }
}

function renderABTestChart(groups) {
    const ctx = document.getElementById('abTestChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: groups.map(g => ALGO_LABELS[g.algorithm_group] || g.algorithm_group),
            datasets: [{
                data: groups.map(g => g.total_clicks),
                backgroundColor: groups.map(g => (CHART_COLORS[g.algorithm_group] || { main: '#999' }).main + 'cc'),
                borderColor: '#fff', borderWidth: 3, hoverOffset: 8,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#565959', usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 13 } } },
                tooltip: {
                    backgroundColor: '#232f3e', titleColor: '#fff', bodyColor: '#ccc', borderColor: '#ff9900', borderWidth: 1, cornerRadius: 6, padding: 10,
                    callbacks: { afterLabel: ctx => `CTR: ${groups[ctx.dataIndex].avg_ctr}%\nSessions: ${groups[ctx.dataIndex].num_sessions}` },
                },
            },
            cutout: '60%',
        },
    });
}

// ========== Admin Dashboard ==========
async function loadAdminStats() {
    const tbody = document.getElementById('adminStatsBody');
    try {
        const stats = await api('/admin/stats');
        
        if (!stats || stats.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No data available yet. Start clicking on recommendations!</td></tr>`;
            return;
        }

        const icons = { collaborative: '🤝', content_based: '📝', hybrid: '🧠', popular: '📊', llm_hybrid: '🤖', also_bought: '🛒' };
        
        let html = '';
        stats.forEach(s => {
            const label = ALGO_LABELS[s.algorithm] || s.algorithm;
            const icon = icons[s.algorithm] || '🔹';
            html += `<tr>
                <td><strong>${icon} ${label}</strong></td>
                <td>${s.total_impressions.toLocaleString()}</td>
                <td>${s.total_clicks.toLocaleString()}</td>
                <td><strong>${s.ctr_pct}%</strong></td>
                <td><strong style="color:var(--amazon-green);">$$${s.total_revenue.toFixed(2)}</strong></td>
            </tr>`;
        });
        
        tbody.innerHTML = html;
        
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:red;">Failed to load admin stats.</td></tr>`;
    }
}

// ========== Marketing Automation ==========
async function generateMarketingEmail() {
    const userId = document.getElementById('marketingUserSelect').value;
    const outputDiv = document.getElementById('marketingEmailOutput');
    
    if (!userId) {
        showToast("Please select a target user.");
        return;
    }
    
    outputDiv.style.display = 'block';
    outputDiv.innerHTML = '<div class="loading-container"><div class="spinner"></div><p>Generating personalized campaign with AI...</p></div>';
    
    try {
        const data = await api(`/admin/generate-email?user_id=${userId}`);
        
        // Simple markdown parsing for the output
        let emailHtml = escapeHtml(data.email)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br/>');
            
        outputDiv.innerHTML = emailHtml;
    } catch (e) {
        outputDiv.innerHTML = '<p style="color:var(--amazon-red);">Failed to generate email.</p>';
    }
}
