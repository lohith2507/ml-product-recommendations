/**
 * Main Application Logic
 * Handles navigation, user selection, data fetching, and global state.
 */

// ========== Global State ==========
const state = {
    currentUser: null,
    currentSection: 'recommendations',
    currentAlgoTab: 'collaborative',
    currentCategory: 'all',
    users: [],
    categories: [],
    isLoading: false,
};

// ========== API Helper ==========
const API_BASE_URL = '/api';

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!response.ok) {
            let detail = `API error: ${response.status}`;
            try {
                const body = await response.json();
                if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
            } catch (_) { /* ignore */ }
            throw new Error(detail);
        }
        return await response.json();
    } catch (error) {
        console.error(`API call failed: ${endpoint}`, error);
        throw error;
    }
}

// ========== Initialization ==========
function applyPlatformLayout() {
    const ua = navigator.userAgent || '';
    const touchMac = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
    const iPad = /iPad/.test(ua) || touchMac;
    const ios = /iPhone|iPod/.test(ua) || iPad;
    const android = /Android/.test(ua);
    const coarse = window.matchMedia('(pointer: coarse)').matches;
    const width = window.innerWidth;
    let form = 'desktop';
    if (width < 700) form = 'phone';
    else if (width < 1100 || (coarse && width < 1400)) form = 'tablet';
    const root = document.documentElement;
    root.dataset.form = form;
    root.dataset.os = ios ? 'ios' : android ? 'android' : 'other';
    root.dataset.pointer = coarse ? 'coarse' : 'fine';
}

function syncOverlay() {
    const chatOpen = document.getElementById('chatPanel')?.classList.contains('open');
    const cartOpen = document.getElementById('cartSidebar')?.classList.contains('open');
    const scrim = document.getElementById('panelScrim');
    const form = document.documentElement.dataset.form;
    const show = (chatOpen || cartOpen) && form !== 'desktop';
    if (scrim) scrim.hidden = !show;
    document.body.classList.toggle('overlay-open', !!(chatOpen || cartOpen));
    document.querySelectorAll('.platform-dock button').forEach(btn => {
        const key = btn.dataset.dock;
        const on = (key === 'chat' && chatOpen) || (key === 'cart' && cartOpen) ||
            (!chatOpen && !cartOpen && key === state.currentSection);
        btn.classList.toggle('active', !!on);
    });
}

function closeOverlayPanels() {
    document.getElementById('chatPanel')?.classList.remove('open');
    document.getElementById('chatToggle')?.classList.remove('is-open');
    const icon = document.querySelector('#chatToggle .chatbot-toggle-icon');
    const label = document.querySelector('#chatToggle .chatbot-toggle-label');
    if (icon) icon.textContent = '💬';
    if (label) label.textContent = 'Chat Agent';
    document.getElementById('cartSidebar')?.classList.remove('open');
    syncOverlay();
}

document.addEventListener('DOMContentLoaded', async () => {
    applyPlatformLayout();
    window.addEventListener('resize', () => {
        applyPlatformLayout();
        syncOverlay();
    });
    window.addEventListener('orientationchange', () => {
        applyPlatformLayout();
        syncOverlay();
    });
    await loadUsers();
    await loadHeroStats();
    await loadCategories();
    await loadCart();
    if (typeof refreshDealAlerts === 'function') {
        refreshDealAlerts();
        setInterval(refreshDealAlerts, 60000);
    }
});

async function proceedToCheckout() {
    try {
        const data = await api(`/cart/${cartUserId()}/checkout`);
        if (!data.primary_url && !(data.items || []).length) {
            showToast('Your cart is empty.', 'error');
            return;
        }
        if (data.primary_url) {
            window.open(data.primary_url, '_blank', 'noopener');
        }
        const extra = (data.items || []).slice(1, 4);
        if (extra.length) {
            showToast(`Opened checkout. ${extra.length} more item link(s) in console.`);
            console.info('Checkout links', data.items);
        } else {
            showToast('Opened store checkout');
        }
    } catch (e) {
        showToast(e.message || 'Checkout handoff failed.', 'error');
    }
}

async function loadIncidents() {
    const box = document.getElementById('incidentList');
    if (!box) return;
    try {
        const data = await api('/incidents');
        const items = data.incidents || [];
        if (!items.length) {
            box.innerHTML = '<p>No incidents yet. Run a scenario above.</p>';
            return;
        }
        box.innerHTML = items.map(item => `
            <article style="padding:12px 0; border-bottom:1px solid var(--border-light);">
                <strong>${escapeHtml(item.title)}</strong>
                <div style="font-size:0.85rem; color:var(--text-medium); margin-top:4px;">
                    ${escapeHtml(item.severity)} · ${escapeHtml(item.status)} · ${escapeHtml(item.category)} · ${Math.round((item.confidence || 0) * 100)}%
                </div>
                <p style="margin-top:6px;">${escapeHtml(item.summary || '')}</p>
                ${(item.evidence || []).slice(0, 3).map(e => `<div style="font-size:0.8rem;">Evidence: ${escapeHtml(e)}</div>`).join('')}
                ${(item.remediations || []).map(r => `<div style="font-size:0.8rem; margin-top:8px;">${escapeHtml(r.title)}: ${escapeHtml(r.status)}${r.result ? '. ' + escapeHtml(r.result) : ''}</div>`).join('')}
                ${item.status === 'pending_approval' ? `<button type="button" class="hero-cta" style="margin-top:8px;" onclick="approveIncident('${item.id}')">Approve fix</button>` : ''}
                ${item.postmortem ? `<pre style="white-space:pre-wrap; font-size:0.78rem; margin-top:8px;">${escapeHtml(item.postmortem)}</pre>` : ''}
            </article>
        `).join('');
    } catch (e) {
        box.innerHTML = '<p>Could not load incidents.</p>';
    }
}

async function runIncidentScenario(name) {
    try {
        await api(`/incidents/scenarios/${name}`, { method: 'POST', body: '{}' });
        switchSection('incidents');
        await loadIncidents();
    } catch (e) {
        showToast(e.message || 'Scenario failed', 'error');
    }
}

async function approveIncident(id) {
    try {
        await api(`/incidents/${id}/approve`, { method: 'POST', body: '{}' });
        await loadIncidents();
    } catch (e) {
        showToast(e.message || 'Approval failed', 'error');
    }
}

function applyUiActions(actions) {
    (actions || []).forEach(action => {
        if (!action || !action.type) return;
        try {
            if (action.type === 'navigate' && action.section) {
                switchSection(action.section);
            } else if (action.type === 'search') {
                const input = document.getElementById('searchInput');
                const category = document.getElementById('searchCategory');
                if (input && action.query != null) input.value = action.query;
                if (category && action.category) category.value = action.category;
                switchSection('products');
                searchProducts();
            } else if (action.type === 'filter_category') {
                switchSection('products');
                filterCategory(action.category || 'all');
            } else if (action.type === 'switch_algo') {
                switchSection('recommendations');
                switchAlgoTab(action.algo || 'collaborative');
            } else if (action.type === 'open_cart') {
                const sidebar = document.getElementById('cartSidebar');
                if (sidebar && !sidebar.classList.contains('open')) toggleCart();
                else loadCart();
            } else if (action.type === 'close_cart') {
                document.getElementById('cartSidebar')?.classList.remove('open');
                if (typeof syncOverlay === 'function') syncOverlay();
            } else if (action.type === 'checkout') {
                proceedToCheckout();
            } else if (action.type === 'show_product' && action.product_id) {
                switchSection('recommendations');
                onProductClick(Number(action.product_id), 'chatbot');
            } else if (action.type === 'summarize_reviews' && action.product_id && typeof summarizeReviews === 'function') {
                summarizeReviews(Number(action.product_id));
            } else if (action.type === 'refresh_cart') {
                loadCart();
            }
        } catch (e) {
            console.error('Site action failed', action, e);
        }
    });
}

async function loadUsers() {
    try {
        const users = await api('/users?limit=50');
        state.users = users;

        const select = document.getElementById('userSelect');
        const marketingSelect = document.getElementById('marketingUserSelect');
        if (select) {
            select.innerHTML = '<option value="">Select a User</option>';
            users.forEach(u => {
                const opt = document.createElement('option');
                opt.value = u.user_id;
                opt.textContent = `${u.username} (User ${u.user_id})`;
                select.appendChild(opt);
                
                if (marketingSelect) {
                    const mOpt = document.createElement('option');
                    mOpt.value = u.user_id;
                    mOpt.textContent = `${u.username} (User ${u.user_id})`;
                    marketingSelect.appendChild(mOpt);
                }
            });
        }

        // Auto-select first user
        if (users.length > 0) {
            select.value = users[0].user_id;
            onUserChange();
        }
    } catch (e) {
        showToast('Failed to load users. Is the server running?', 'error');
    }
}

async function loadHeroStats() {
    try {
        const products = await api('/products?limit=1');
        document.getElementById('heroProducts').textContent = (products.total || 0).toLocaleString();

        const users = state.users;
        document.getElementById('heroUsers').textContent = users.length || '-';

        const totalReviews = users.reduce((sum, u) => sum + (u.review_count || 0), 0);
        document.getElementById('heroReviews').textContent = totalReviews.toLocaleString() || '-';
    } catch (e) { /* Stats will show defaults */ }
}

async function loadCategories() {
    try {
        const catData = await api('/products/categories/list');
        const cats = catData.map(c => c.category).sort();
        state.categories = cats;

        // Populate search dropdown
        const searchCat = document.getElementById('searchCategory');
        cats.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.textContent = cat;
            searchCat.appendChild(opt);
        });

        // Populate product filter buttons
        const tabContainer = document.querySelector('#section-products .tab-container');
        cats.slice(0, 10).forEach(cat => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn';
            btn.dataset.category = cat;
            btn.textContent = cat.length > 20 ? cat.substring(0, 18) + '...' : cat;
            btn.title = cat;
            btn.onclick = () => filterCategory(cat);
            tabContainer.appendChild(btn);
        });
    } catch (e) { /* Silently fail */ }
}

// ========== Navigation ==========
function switchSection(section) {
    state.currentSection = section;

    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.toggle('active', link.dataset.section === section);
    });

    document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
    document.getElementById(`section-${section}`).classList.add('active');
    if (document.documentElement.dataset.form === 'phone') {
        document.getElementById('chatPanel')?.classList.remove('open');
        document.getElementById('cartSidebar')?.classList.remove('open');
    }
    if (typeof syncOverlay === 'function') syncOverlay();

    if (section === 'products' && !document.getElementById('productsContainer').dataset.loaded) {
        loadProducts();
    }
    if (section === 'analytics') loadAnalytics();
    if (section === 'abtest') loadABTestResults();
    if (section === 'admin') loadAdminStats();
    if (section === 'incidents' && typeof loadIncidents === 'function') loadIncidents();
}

// ========== User Selection ==========
function onUserChange() {
    const select = document.getElementById('userSelect');
    const userId = parseInt(select.value);
    const previous = state.currentUser;
    if (!userId) {
        state.currentUser = null;
        return;
    }
    if (previous && previous !== userId && typeof snapshotChatForUser === 'function') {
        snapshotChatForUser(previous);
    }
    state.currentUser = userId;
    loadRecommendations();
    loadCart();
    if (typeof loadChatForUser === 'function') loadChatForUser(userId);
    if (typeof refreshDealAlerts === 'function') refreshDealAlerts();
}

// ========== Recommendation Tabs ==========
function switchAlgoTab(algo) {
    state.currentAlgoTab = algo;
    document.querySelectorAll('.tab-btn[data-algo]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.algo === algo);
    });

    const titles = {
        'collaborative': 'Recommended for You (Collaborative Filtering)',
        'content': 'Based on Your Interests (Content-Based)',
        'hybrid': 'AI Picks for You (Hybrid Intelligence)',
    };
    document.getElementById('recSectionTitle').textContent = titles[algo] || 'Recommended for You';
    loadRecommendations();
}

async function loadRecommendations() {
    if (!state.currentUser) {
        document.getElementById('recommendationsContainer').innerHTML = `
            <div class="empty-state" style="width:100%;">
                <div class="empty-icon">👤</div>
                <p>Select a user from the top bar to see personalized recommendations</p>
            </div>`;
        return;
    }

    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = `
        <div class="loading-container" style="width:100%;">
            <div class="spinner"></div>
            <p class="loading-text">Finding the best products for you...</p>
        </div>`;

    try {
        const data = await api(`/recommend/${state.currentAlgoTab}/${state.currentUser}?n=20`);
        renderRecommendations(data.recommendations, container);

        if (data.recommendations.length > 0) {
            loadAlsoBought(data.recommendations[0].product_id);
        }
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state" style="width:100%;">
                <div class="empty-icon">⚠️</div>
                <p>Failed to load recommendations. Make sure models are trained.</p>
            </div>`;
    }
}

async function loadAlsoBought(productId) {
    const section = document.getElementById('alsoBoughtSection');
    const container = document.getElementById('alsoBoughtContainer');

    try {
        const data = await api(`/recommend/also-bought/${productId}?n=10`);
        if (data.recommendations.length > 0) {
            section.style.display = 'block';
            renderRecommendations(data.recommendations, container);
        } else {
            section.style.display = 'none';
        }
    } catch (e) { section.style.display = 'none'; }
}

// ========== Products Catalog ==========
async function loadProducts(category = 'all') {
    const container = document.getElementById('productsContainer');
    container.innerHTML = `
        <div class="loading-container" style="grid-column: 1/-1;">
            <div class="spinner"></div>
            <p class="loading-text">Loading products...</p>
        </div>`;

    try {
        let endpoint = '/products?limit=40';
        if (category && category !== 'all') {
            endpoint += `&category=${encodeURIComponent(category)}`;
        }
        const data = await api(endpoint);
        container.dataset.loaded = 'true';
        renderProductCatalog(data.products, container);
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1/-1;">
                <div class="empty-icon">📦</div>
                <p>Failed to load products.</p>
            </div>`;
    }
}

function filterCategory(category) {
    state.currentCategory = category;
    document.querySelectorAll('.tab-btn[data-category]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.category === category);
    });
    loadProducts(category);
}

// ========== Search ==========
async function searchProducts() {
    const query = document.getElementById('searchInput').value.trim();
    const category = document.getElementById('searchCategory').value;

    if (!query && !category) return;

    // Switch to products section
    switchSection('products');

    const container = document.getElementById('productsContainer');
    container.innerHTML = `
        <div class="loading-container" style="grid-column: 1/-1;">
            <div class="spinner"></div>
            <p class="loading-text">Semantic Searching...</p>
        </div>`;

    try {
        let endpoint = '';
        if (query) {
            // Use semantic search for text queries
            endpoint = `/search/semantic?q=${encodeURIComponent(query)}&limit=50`;
        } else {
            // Use standard filter for category only
            endpoint = `/products?limit=50&category=${encodeURIComponent(category)}`;
        }
        
        const data = await api(endpoint);
        
        // If category is selected but we used semantic search, filter results locally
        let finalProducts = data.products;
        if (query && category && category !== 'all') {
            finalProducts = finalProducts.filter(p => p.category === category);
        }
        
        renderProductCatalog(finalProducts, container);

        if (finalProducts.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1;">
                    <div class="empty-icon">🔍</div>
                    <p>No products found for "${escapeHtml(query || category)}"</p>
                    <p style="font-size:0.85rem; margin-top:8px;">Check out some popular suggestions instead:</p>
                </div>`;
            
            // Fetch some popular products as a fallback
            try {
                const fallbackData = await api('/products?limit=8');
                if (fallbackData.products && fallbackData.products.length > 0) {
                    const fallbackContainer = document.createElement('div');
                    fallbackContainer.className = 'product-grid';
                    fallbackContainer.style = 'grid-column: 1/-1; margin-top: 16px;';
                    container.appendChild(fallbackContainer);
                    
                    // Render suggestions using the existing helper, but to the inner container
                    const html = fallbackData.products.map(product => {
                        const price = parseFloat(product.price) || 0;
                        const listPrice = parseFloat(product.list_price) || 0;
                        const rating = parseFloat(product.avg_rating) || 0;
                        const ratingCount = product.rating_count || 0;
                        const isBestSeller = product.is_best_seller === 1;
                        const img = product.image_url || `https://via.placeholder.com/300x300?text=${encodeURIComponent(product.category)}`;

                        let badge = isBestSeller ? '<div class="best-seller-badge">Best Seller</div>' : '';

                        let priceHtml = `
                            <div class="price-current">
                                <span class="price-symbol">$</span><span class="price-whole">${Math.floor(price)}</span><span class="price-fraction">${(price % 1).toFixed(2).substring(2)}</span>
                            </div>`;

                        if (listPrice > price && price > 0) {
                            priceHtml += `
                                <div style="display:flex; align-items:center;">
                                    <span class="price-list">List: $${listPrice.toFixed(2)}</span>
                                </div>`;
                        }

                        return `
                        <div class="product-card" onclick="onProductClick(${product.product_id}, 'popular')">
                            <div class="product-image">
                                ${badge}
                                <img src="${escapeHtml(img)}" alt="${escapeHtml(product.title)}" loading="lazy" onerror="this.src='https://via.placeholder.com/300x300?text=No+Image'">
                            </div>
                            <div class="product-body">
                                <div class="product-title" title="${escapeHtml(product.title)}">${escapeHtml(product.title)}</div>
                                <div class="product-rating">
                                    <span class="stars-display">★★★★★</span>
                                    <span class="rating-count">${ratingCount.toLocaleString()}</span>
                                </div>
                                <div class="product-meta">
                                    ${priceHtml}
                                </div>
                            </div>
                        </div>`;
                    }).join('');
                    fallbackContainer.innerHTML = html;
                }
            } catch (fallbackError) {
                console.error("Failed to load fallback suggestions", fallbackError);
            }
        }
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1/-1;">
                <div class="empty-icon">⚠️</div>
                <p>Search failed. Please try again.</p>
            </div>`;
    }
}

// ========== Visual Search Logic ==========
async function performVisualSearch(inputElement) {
    if (!inputElement.files || inputElement.files.length === 0) return;
    
    const file = inputElement.files[0];
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file.');
        return;
    }
    
    switchSection('search');
    const container = document.getElementById('searchResultsContainer');
    container.innerHTML = `
        <div class="loading-container" style="grid-column: 1/-1;">
            <div class="spinner"></div>
            <p class="loading-text">Analyzing image with Groq Vision...</p>
        </div>`;
        
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/search/visual?limit=12`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Visual search failed");
        
        const data = await response.json();
        const finalProducts = data.products || [];
        
        // Reset file input
        inputElement.value = '';
        
        document.getElementById('searchQueryDisplay').innerText = `Visual Search Results`;
        document.getElementById('searchCountDisplay').innerText = `${finalProducts.length} items found based on image`;
        
        renderProductCatalog(finalProducts, container);
        
        if (finalProducts.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1;">
                    <div class="empty-icon">🔍</div>
                    <p>No visually similar products found.</p>
                </div>`;
        }
    } catch (e) {
        console.error(e);
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1/-1;">
                <div class="empty-icon">⚠️</div>
                <p>Visual search failed. Please try again.</p>
            </div>`;
    }
}

// ========== Toast Notifications ==========
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> <span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 3000);
}

// ========== Click Tracking ==========
async function trackClick(productId, algorithm) {
    if (!state.currentUser) return;
    try {
        await api('/simulate/click', {
            method: 'POST',
            body: JSON.stringify({
                user_id: state.currentUser,
                product_id: productId,
                algorithm: algorithm,
            }),
        });
    } catch (e) { /* Silent fail for tracking */ }
}

// ========== Shopping Cart Logic ==========
let cart = { items: [], item_count: 0, subtotal: 0, savings: 0 };

function cartUserId() {
    return state.currentUser || 1;
}

function applyCartUpdate(updatedCart) {
    cart = updatedCart;
    document.getElementById('cartCountBadge').innerText = cart.item_count;
    const savingsEl = document.getElementById('cartSavingsTotal');
    if (savingsEl) savingsEl.innerText = `$${(cart.savings || 0).toFixed(2)}`;
    if (document.getElementById('cartSidebar').classList.contains('open')) {
        renderCart();
    }
}

async function loadCart() {
    try {
        applyCartUpdate(await api(`/cart/${cartUserId()}`));
    } catch (e) {
        showToast('Could not load your cart.', 'error');
    }
}

function toggleCart() {
    const sidebar = document.getElementById('cartSidebar');
    const open = !sidebar.classList.contains('open');
    sidebar.classList.toggle('open', open);
    if (open) {
        document.getElementById('chatPanel')?.classList.remove('open');
        document.getElementById('chatToggle')?.classList.remove('is-open');
        const icon = document.querySelector('#chatToggle .chatbot-toggle-icon');
        const label = document.querySelector('#chatToggle .chatbot-toggle-label');
        if (icon) icon.textContent = '💬';
        if (label) label.textContent = 'Chat Agent';
        renderCart();
    }
    if (typeof syncOverlay === 'function') syncOverlay();
}

async function addToCart(productId, title) {
    try {
        applyCartUpdate(await api(`/cart/${cartUserId()}/items`, {
            method: 'POST',
            body: JSON.stringify({ product_id: productId, quantity: 1 }),
        }));
    } catch (e) {
        showToast('Could not add that item to your cart.', 'error');
        return;
    }

    showToast(`Added ${title} to cart!`);

    if (!document.getElementById('cartSidebar').classList.contains('open')) {
        toggleCart();
    }
}

async function removeFromCart(productId) {
    const pid = Number(productId);
    if (!Number.isFinite(pid)) {
        showToast('Invalid product to remove.', 'error');
        return;
    }
    try {
        const updated = await api(`/cart/${cartUserId()}/items/${pid}`, { method: 'DELETE' });
        applyCartUpdate(updated);
        renderCart();
        showToast('Removed from cart');
    } catch (e) {
        showToast(e.message || 'Could not remove that item.', 'error');
    }
}

async function renderCart() {
    const cartItemsDiv = document.getElementById('cartItems');
    const subtotalEl = document.getElementById('cartSubtotal');
    const savingsEl = document.getElementById('cartSavingsTotal');

    if (cart.items.length === 0) {
        cartItemsDiv.innerHTML = '<div class="empty-state"><p>Your cart is empty.</p></div>';
        subtotalEl.innerText = '$0.00';
        if (savingsEl) savingsEl.innerText = '$0.00';
        document.getElementById('cartRecsContainer').innerHTML = '<p class="cart-recs-empty">Add items to see recommendations.</p>';
        return;
    }

    let html = '';
    cart.items.forEach(item => {
        const price = Number(item.price) || 0;
        const pid = Number(item.product_id);
        html += `
            <div class="cart-line-item">
                <img src="${item.image_url || 'https://via.placeholder.com/50'}" alt="" />
                <div class="cart-line-body">
                    <div class="cart-line-title">${escapeHtml(item.title)}</div>
                    <div class="cart-line-meta">Qty ${item.quantity} · <strong>$${price.toFixed(2)}</strong></div>
                </div>
                <button type="button" class="cart-remove" data-product-id="${pid}">Remove</button>
            </div>
        `;
    });

    cartItemsDiv.innerHTML = html;
    cartItemsDiv.querySelectorAll('.cart-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            removeFromCart(btn.getAttribute('data-product-id'));
        });
    });
    subtotalEl.innerText = `$${(cart.subtotal || 0).toFixed(2)}`;
    if (savingsEl) savingsEl.innerText = `$${(cart.savings || 0).toFixed(2)}`;

    fetchCartRecommendations();
}

async function fetchCartRecommendations() {
    const recsContainer = document.getElementById('cartRecsContainer');
    if (!recsContainer) return;
    recsContainer.innerHTML = '<p class="cart-recs-empty">Loading recommendations...</p>';

    try {
        const productIds = cart.items.map(i => Number(i.product_id)).filter(Number.isFinite);
        let recs = [];
        try {
            const data = await api(`/recommend/cart?n=4`, {
                method: 'POST',
                body: JSON.stringify({ product_ids: productIds }),
            });
            recs = data.recommendations || [];
        } catch (_) {
            if (productIds.length) {
                const data = await api(`/recommend/also-bought/${productIds[0]}?n=4`);
                recs = data.recommendations || data.products || (Array.isArray(data) ? data : []);
            }
        }

        if (!recs.length) {
            recsContainer.innerHTML = '<p class="cart-recs-empty">No recommendations available.</p>';
            return;
        }

        let html = '';
        recs.forEach(rec => {
            const price = Number(rec.price) || 0;
            html += `
                <div class="cart-rec-row" onclick="onProductClick(${rec.product_id}, 'also_bought')">
                    <img src="${rec.image_url || 'https://via.placeholder.com/40'}" alt="" />
                    <div class="cart-rec-body">
                        <div class="cart-rec-title">${escapeHtml(rec.title || '')}</div>
                        <div class="cart-rec-price">$${price.toFixed(2)}</div>
                    </div>
                </div>
            `;
        });
        recsContainer.innerHTML = html;
    } catch (e) {
        console.error(e);
        recsContainer.innerHTML = '<p class="cart-recs-error">Recommendations unavailable right now.</p>';
    }
}
