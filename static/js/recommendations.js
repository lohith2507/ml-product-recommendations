/**
 * Product Card Rendering — Amazon Style
 * Renders product cards with real images, Amazon-style pricing,
 * best seller badges, star ratings, and algorithm badges.
 */

/**
 * Render recommendation product cards in a carousel container.
 */
function renderRecommendations(recommendations, container) {
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="width:100%;">
                <div class="empty-icon">🔍</div>
                <p>No recommendations available for this user.</p>
            </div>`;
        return;
    }
    container.innerHTML = recommendations.map(rec => createProductCard(rec)).join('');
}

/**
 * Render product catalog cards in a grid.
 */
function renderProductCatalog(products, container) {
    if (!products || products.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1/-1;">
                <div class="empty-icon">📦</div>
                <p>No products found in this category.</p>
            </div>`;
        return;
    }
    container.innerHTML = products.map(product => createProductCard(product, true)).join('');
}

/**
 * Create a single Amazon-style product card HTML.
 */
function createProductCard(product, isCatalog = false) {
    const algo = product.algorithm || '';
    const algoLabel = getAlgoLabel(algo);
    const price = parseFloat(product.price) || 0;
    const listPrice = parseFloat(product.list_price) || 0;
    const isBestSeller = product.is_best_seller;
    const boughtLastMonth = product.bought_last_month || 0;
    const avgRating = parseFloat(product.avg_rating) || 0;
    const ratingCount = product.rating_count || 0;
    const imgUrl = product.image_url || '';
    const title = product.title || 'Untitled Product';

    // Image
    let imageHtml;
    if (imgUrl && imgUrl.startsWith('http')) {
        imageHtml = `<img src="${escapeHtml(imgUrl)}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.style.display='none'; this.parentElement.querySelector('.category-icon').style.display='flex';">
                     <span class="category-icon" style="display:none;">📦</span>`;
    } else {
        imageHtml = `<span class="category-icon">📦</span>`;
    }

    // Best Seller badge
    let bestSellerBadge = '';
    if (isBestSeller) {
        bestSellerBadge = `<div class="best-seller-badge">Best Seller</div>`;
    }

    // Algorithm badge
    let badgeHtml = '';
    if (algo && !isCatalog) {
        const badgeClass = algo.replace(/[^a-z_]/g, '');
        badgeHtml = `<span class="algo-badge ${badgeClass}">${algoLabel}</span>`;
    }

    // Star rating display
    const stars = renderStars(avgRating);
    let ratingHtml = `
        <div class="product-rating">
            <span class="stars-display">${stars}</span>
            <span class="rating-count">${formatCount(ratingCount)}</span>
        </div>`;

    // Price display (Amazon-style)
    let priceHtml = formatAmazonPrice(price, listPrice);

    // Bought count
    let boughtHtml = '';
    if (boughtLastMonth > 0) {
        boughtHtml = `<div class="bought-count">${formatBoughtCount(boughtLastMonth)} bought in past month</div>`;
    }

    // Score display (for recommendations)
    let scoreHtml = '';
    if (!isCatalog) {
        if (product.predicted_rating) {
            scoreHtml = `<div class="product-score">Predicted: ⭐ ${product.predicted_rating.toFixed(2)}</div>`;
        } else if (product.similarity_score) {
            scoreHtml = `<div class="product-score">Match: ${(product.similarity_score * 100).toFixed(1)}%</div>`;
        } else if (product.confidence) {
            scoreHtml = `<div class="product-score">Confidence: ${(product.confidence * 100).toFixed(0)}%</div>`;
        } else if (product.co_purchase_count) {
            scoreHtml = `<div class="product-score">${product.co_purchase_count} customers also bought this</div>`;
        }
    }

    // Explanation
    let explanationHtml = '';
    if (product.explanation && !isCatalog) {
        explanationHtml = `<div class="product-explanation">💡 ${escapeHtml(product.explanation)}</div>`;
    }

    // Actions
    let actionsHtml = `
        <div class="product-actions" style="margin-top: 8px; display: flex; gap: 6px;">
            <button class="action-btn" onclick="event.stopPropagation(); addToCart(${product.product_id}, '${escapeHtml(title).replace(/'/g, "\\'")}', ${price}, '${escapeHtml(product.image_url || '')}')" title="Add to Cart">
                🛒 Add
            </button>
            <button class="action-btn" onclick="event.stopPropagation(); summarizeReviews(${product.product_id})" title="AI Review Summary">
                ✨ Summarize
            </button>
        </div>`;

    return `
        <div class="product-card" onclick="onProductClick(${product.product_id}, '${algo}')">
            <div class="product-image">
                ${imageHtml}
                ${bestSellerBadge}
                ${badgeHtml}
            </div>
            <div class="product-body">
                <div class="product-title">${escapeHtml(title)}</div>
                ${ratingHtml}
                <div class="product-meta">
                    ${priceHtml}
                </div>
                ${boughtHtml}
                ${scoreHtml}
                ${explanationHtml}
                ${actionsHtml}
            </div>
        </div>`;
}

/**
 * Handle product card click.
 */
function onProductClick(productId, algorithm) {
    if (algorithm) trackClick(productId, algorithm);
    loadAlsoBought(productId);

    if (state.currentSection === 'recommendations') {
        setTimeout(() => {
            const section = document.getElementById('alsoBoughtSection');
            if (section && section.style.display !== 'none') {
                section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
    }
}

// ========== Formatting Helpers ==========

function getAlgoLabel(algo) {
    const labels = {
        'collaborative': 'Collaborative',
        'collaborative_popular': 'Popular',
        'content_based': 'Content-Based',
        'content_similar': 'Similar',
        'content_popular': 'Trending',
        'llm_hybrid': 'AI Hybrid',
        'hybrid_cf': 'Hybrid (CF)',
        'hybrid_cb': 'Hybrid (CB)',
        'also_bought': 'Also Bought',
        'chatbot': 'AI Pick',
    };
    return labels[algo] || algo;
}

// ========== AI Review Summarization ==========
async function summarizeReviews(productId) {
    const modal = document.getElementById('reviewModal');
    const modalBody = document.getElementById('reviewModalBody');
    
    modal.style.display = 'block';
    modalBody.innerHTML = '<div class="loading-container"><div class="spinner"></div><p>Reading and summarizing reviews...</p></div>';
    
    try {
        const data = await api(`/products/${productId}/summary`);
        
        let html = `<p style="font-size:0.95rem; line-height:1.5; color:var(--text-dark); margin-bottom:16px;">${escapeHtml(data.summary)}</p>`;
        
        if (data.pros && data.pros.length > 0) {
            html += `<h4 style="color:var(--amazon-green); margin-bottom:8px;">✅ Pros</h4>
                     <ul style="margin-left:20px; font-size:0.85rem; margin-bottom:16px;">
                        ${data.pros.map(p => `<li>${escapeHtml(p)}</li>`).join('')}
                     </ul>`;
        }
        
        if (data.cons && data.cons.length > 0) {
            html += `<h4 style="color:var(--amazon-red); margin-bottom:8px;">❌ Cons</h4>
                     <ul style="margin-left:20px; font-size:0.85rem;">
                        ${data.cons.map(c => `<li>${escapeHtml(c)}</li>`).join('')}
                     </ul>`;
        }
        
        modalBody.innerHTML = html;
        
    } catch (e) {
        modalBody.innerHTML = `<div class="empty-state"><p style="color:var(--amazon-red);">Failed to generate summary.</p></div>`;
    }
}

/**
 * Amazon-style price display with optional list price strikethrough.
 */
function formatAmazonPrice(price, listPrice) {
    if (!price || price <= 0) {
        return `<span class="price-current"><span class="price-symbol">$</span><span class="price-whole">0</span><span class="price-fraction">00</span></span>`;
    }

    const whole = Math.floor(price);
    const fraction = Math.round((price - whole) * 100).toString().padStart(2, '0');

    let html = `<span class="price-current"><span class="price-symbol">$</span><span class="price-whole">${whole}</span><span class="price-fraction">${fraction}</span></span>`;

    if (listPrice > price) {
        const discount = Math.round((1 - price / listPrice) * 100);
        html += `<span class="price-list">$${listPrice.toFixed(2)}</span>`;
        if (discount > 0) {
            html += `<span class="price-discount">(-${discount}%)</span>`;
        }
    }

    return html;
}

function renderStars(rating) {
    const full = Math.floor(rating);
    const half = rating % 1 >= 0.3;
    let stars = '';
    for (let i = 0; i < full; i++) stars += '★';
    if (half) stars += '★';
    for (let i = stars.length; i < 5; i++) stars += '☆';
    return stars;
}

function formatCount(count) {
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
}

function formatBoughtCount(count) {
    if (count >= 10000) return `${Math.round(count / 1000)}K+`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K+`;
    if (count >= 100) return `${Math.round(count / 100) * 100}+`;
    return `${count}+`;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
