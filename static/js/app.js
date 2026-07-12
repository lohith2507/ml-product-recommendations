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
async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`/api${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API call failed: ${endpoint}`, error);
        throw error;
    }
}

// ========== Initialization ==========
document.addEventListener('DOMContentLoaded', async () => {
    await loadUsers();
    await loadHeroStats();
    await loadCategories();
});

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
        document.getElementById('heroUsers').textContent = users.length || '—';

        const totalReviews = users.reduce((sum, u) => sum + (u.review_count || 0), 0);
        document.getElementById('heroReviews').textContent = totalReviews.toLocaleString() || '—';
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

    if (section === 'products' && !document.getElementById('productsContainer').dataset.loaded) {
        loadProducts();
    }
    if (section === 'analytics') loadAnalytics();
    if (section === 'abtest') loadABTestResults();
    if (section === 'admin') loadAdminStats();
}

// ========== User Selection ==========
function onUserChange() {
    const select = document.getElementById('userSelect');
    const userId = parseInt(select.value);
    if (!userId) { state.currentUser = null; return; }
    state.currentUser = userId;
    loadRecommendations();
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
let cart = [];

function toggleCart() {
    const sidebar = document.getElementById('cartSidebar');
    if (sidebar.style.right === '0px') {
        sidebar.style.right = '-470px';
    } else {
        sidebar.style.right = '0px';
        renderCart();
    }
}

function addToCart(productId, title, price, imageUrl) {
    const existingItem = cart.find(i => i.product_id === productId);
    if (existingItem) {
        existingItem.quantity += 1;
    } else {
        cart.push({ product_id: productId, title, price, imageUrl, quantity: 1 });
    }
    
    // Update badge
    const badge = document.getElementById('cartCountBadge');
    badge.innerText = cart.reduce((sum, item) => sum + item.quantity, 0);
    
    showToast(`Added ${title} to cart!`);
    
    // Open sidebar automatically
    const sidebar = document.getElementById('cartSidebar');
    if (sidebar.style.right !== '0px') {
        toggleCart();
    } else {
        renderCart();
    }
}

function removeFromCart(productId) {
    cart = cart.filter(i => i.product_id !== productId);
    
    // Update badge
    const badge = document.getElementById('cartCountBadge');
    badge.innerText = cart.reduce((sum, item) => sum + item.quantity, 0);
    
    renderCart();
}

async function renderCart() {
    const cartItemsDiv = document.getElementById('cartItems');
    const subtotalEl = document.getElementById('cartSubtotal');
    
    if (cart.length === 0) {
        cartItemsDiv.innerHTML = '<div class="empty-state"><p>Your cart is empty.</p></div>';
        subtotalEl.innerText = '$0.00';
        document.getElementById('cartRecsContainer').innerHTML = '<p style="font-size:0.75rem; color:var(--text-light);">Add items to cart to see recommendations.</p>';
        return;
    }
    
    let html = '';
    let subtotal = 0;
    
    cart.forEach(item => {
        const itemTotal = item.price * item.quantity;
        subtotal += itemTotal;
        
        html += `
            <div style="display:flex; gap:12px; background:white; padding:12px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                <img src="${item.imageUrl || 'https://via.placeholder.com/50'}" style="width:50px; height:50px; object-fit:contain;" />
                <div style="flex:1;">
                    <div style="font-weight:500; font-size:0.9rem; line-height:1.2; margin-bottom:4px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${escapeHtml(item.title)}</div>
                    <div style="font-size:0.85rem; color:var(--text-medium);">Qty: ${item.quantity} | <span style="color:var(--amazon-red); font-weight:bold;">$${item.price.toFixed(2)}</span></div>
                </div>
                <button onclick="removeFromCart(${item.product_id})" style="background:none; border:none; color:var(--text-light); cursor:pointer; font-size:1.2rem; align-self:center;">🗑️</button>
            </div>
        `;
    });
    
    cartItemsDiv.innerHTML = html;
    subtotalEl.innerText = `$${subtotal.toFixed(2)}`;
    
    // Fetch Frequently Bought Together
    fetchCartRecommendations();
}

async function fetchCartRecommendations() {
    const recsContainer = document.getElementById('cartRecsContainer');
    recsContainer.innerHTML = '<p style="font-size:0.75rem; color:var(--text-light);">Loading recommendations...</p>';
    
    try {
        const productIds = cart.map(i => i.product_id);
        const res = await fetch(`${API_BASE_URL}/recommend/cart?n=4`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_ids: productIds })
        });
        
        if (!res.ok) throw new Error("Failed to fetch cart recs");
        
        const data = await res.json();
        
        if (!data.recommendations || data.recommendations.length === 0) {
            recsContainer.innerHTML = '<p style="font-size:0.75rem; color:var(--text-light);">No recommendations available.</p>';
            return;
        }
        
        let html = '';
        data.recommendations.forEach(rec => {
            html += `
                <div style="display:flex; gap:8px; align-items:center; background:white; padding:8px; border-radius:4px; border:1px solid var(--border-light); cursor:pointer;" onclick="onProductClick(${rec.product_id}, 'also_bought')">
                    <img src="${rec.image_url || 'https://via.placeholder.com/40'}" style="width:40px; height:40px; object-fit:contain;" />
                    <div style="flex:1;">
                        <div style="font-size:0.8rem; line-height:1.2; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${escapeHtml(rec.title)}</div>
                        <div style="color:var(--amazon-red); font-weight:bold; font-size:0.8rem;">$${rec.price.toFixed(2)}</div>
                    </div>
                </div>
            `;
        });
        
        recsContainer.innerHTML = html;
    } catch (e) {
        console.error(e);
        recsContainer.innerHTML = '<p style="font-size:0.75rem; color:var(--amazon-red);">Could not load recommendations.</p>';
    }
}
