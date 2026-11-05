/**
 * AI Chatbot Widget
 * Handles the floating chat panel, message sending, and product rendering in chat.
 */

let chatHistory = [];
const chatsByUser = {};
let chatLoadToken = 0;
let isChatSending = false;

function selectedMemberLabel() {
    const select = document.getElementById('userSelect');
    const text = select?.selectedOptions?.[0]?.textContent || '';
    return text.replace(/\s*\(User\s+\d+\)\s*$/, '').trim();
}

function snapshotChatForUser(userId) {
    if (!userId) return;
    chatsByUser[userId] = chatHistory.slice();
}

function renderChatTranscript(memberLabel) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    container.innerHTML = '';
    const who = memberLabel || 'this member';
    const header = document.querySelector('#chatPanel .chat-info p');
    if (header) header.textContent = `● ${who} · their cart, prefs, and past chats`;
    if (!chatHistory.length) {
        addChatMessage(
            `Hi ${who}. This chat is only yours: your cart, wishlist, and saved chats. Ask for deals or say "like last time".`,
            'assistant'
        );
        return;
    }
    chatHistory.forEach(msg => {
        addChatMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant');
    });
}

async function loadChatForUser(userId) {
    const token = ++chatLoadToken;
    const label = selectedMemberLabel() || `Member ${userId}`;
    if (chatsByUser[userId]) {
        chatHistory = chatsByUser[userId].slice();
        renderChatTranscript(label);
        return;
    }
    chatHistory = [];
    const container = document.getElementById('chatMessages');
    if (container) {
        container.innerHTML = `<div class="chat-message assistant">Loading ${escapeHtml(label)}'s chat...</div>`;
    }
    try {
        const data = await api(`/chat/thread/${userId}`);
        if (token !== chatLoadToken || state.currentUser !== userId) return;
        chatHistory = (data.messages || []).filter(m => m && m.content && (m.role === 'user' || m.role === 'assistant'));
    } catch (e) {
        if (token !== chatLoadToken || state.currentUser !== userId) return;
        chatHistory = [];
    }
    chatsByUser[userId] = chatHistory.slice();
    renderChatTranscript(label);
}

function rememberChatTurn(userId, userText, assistantText) {
    const bucket = (state.currentUser === userId ? chatHistory : (chatsByUser[userId] || [])).slice();
    if (userText) bucket.push({ role: 'user', content: userText });
    if (assistantText) bucket.push({ role: 'assistant', content: assistantText });
    chatsByUser[userId] = bucket;
    if (state.currentUser === userId) chatHistory = bucket;
}

/**
 * Toggle the chatbot panel open/closed without wiping toggle markup.
 */
function toggleChat() {
    const panel = document.getElementById('chatPanel');
    const toggle = document.getElementById('chatToggle');
    const open = !panel.classList.contains('open');
    panel.classList.toggle('open', open);
    toggle.classList.toggle('is-open', open);
    const label = toggle.querySelector('.chatbot-toggle-label');
    const icon = toggle.querySelector('.chatbot-toggle-icon');
    if (icon) icon.textContent = open ? '✕' : '💬';
    if (label) label.textContent = open ? 'Close' : 'Chat Agent';
    if (open) {
        const cart = document.getElementById('cartSidebar');
        if (cart) cart.classList.remove('open');
        document.getElementById('chatInput')?.focus();
    }
    if (typeof syncOverlay === 'function') syncOverlay();
}

function openChatAgent() {
    const panel = document.getElementById('chatPanel');
    if (!panel.classList.contains('open')) toggleChat();
    closeLabsMenu();
}

function closeChatAgent() {
    const panel = document.getElementById('chatPanel');
    if (panel.classList.contains('open')) toggleChat();
}

function sendSuggestion(text) {
    const input = document.getElementById('chatInput');
    input.value = text;
    sendChat();
}

function toggleLabsMenu(event) {
    event.stopPropagation();
    document.getElementById('labsMenu')?.classList.toggle('open');
}

function closeLabsMenu() {
    document.getElementById('labsMenu')?.classList.remove('open');
}

document.addEventListener('click', () => closeLabsMenu());

/**
 * Send a message to the AI chatbot.
 */
async function sendChat() {
    if (isChatSending) return;

    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    const userId = state.currentUser || 1;

    // Add user message to UI
    addChatMessage(message, 'user');
    input.value = '';

    // Show typing indicator
    const typingId = addTypingIndicator();

    isChatSending = true;
    document.getElementById('chatSendBtn').disabled = true;

    try {
        // Build conversation history for context
        const apiHistory = chatHistory.slice(-6).map(msg => ({
            role: msg.role,
            content: msg.content,
        }));

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                message: message,
                conversation_history: apiHistory,
            }),
        });

        if (!response.ok) {
            let data = {};
            try { data = await response.json(); } catch (_) { data = {}; }
            removeTypingIndicator(typingId);
            addChatMessage(data.response || 'I could not finish that. Try a shorter request.', 'assistant', data.products || []);
            if (typeof applyUiActions === 'function') applyUiActions(data.ui_actions);
            return;
        }

        const data = await response.json();
        removeTypingIndicator(typingId);
        rememberChatTurn(userId, message, data.response);
        if (state.currentUser !== userId) return;

        addChatMessage(data.response, 'assistant', data.products, data.pending_actions, {
            message,
            tools_used: data.tools_used,
            thread_id: data.thread_id,
        });
        if (typeof applyUiActions === 'function') applyUiActions(data.ui_actions);

        if (state.currentUser === userId && data.active_context && data.active_context.length > 0) {
            updateActiveContext(data.active_context);
        }

    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage('I could not finish that. Try a shorter request, like a search or a page to open.', 'assistant');
    } finally {
        isChatSending = false;
        document.getElementById('chatSendBtn').disabled = false;
        input.focus();
    }
}

function updateActiveContext(tags) {
    const contextContainer = document.getElementById('chatActiveContext');
    const tagsContainer = document.getElementById('chatContextTags');
    if (!contextContainer || !tagsContainer) return;

    if (!tags || tags.length === 0) {
        contextContainer.classList.remove('visible');
        tagsContainer.innerHTML = '';
        return;
    }

    contextContainer.classList.add('visible');
    tagsContainer.innerHTML = tags.map(tag =>
        `<span class="chat-context-tag">${escapeHtml(tag)}</span>`
    ).join('');
}

async function clearChatMemory() {
    const userId = state.currentUser;
    chatHistory = [];
    if (userId) {
        chatsByUser[userId] = [];
        try {
            await api(`/chat/memory/${userId}`, { method: 'DELETE' });
        } catch (e) {
            console.error('Clear memory failed', e);
        }
    }
    renderChatTranscript(selectedMemberLabel());
    updateActiveContext([]);
    if (typeof showToast === 'function') showToast('Chat memory cleared');
}

/**
 * Add a message bubble to the chat panel.
 */
function addChatMessage(text, role, products = [], pendingActions = [], feedbackMeta = null) {
    const container = document.getElementById('chatMessages');

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${role}`;

    // Escape first, then allow only **bold** and line breaks from the model's text.
    let html = escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    // Add product cards if present
    if (products && products.length > 0) {
        html += '<div class="chat-products">';
        products.forEach(product => {
            const price = parseFloat(product.price) || 0;
            const rating = parseFloat(product.avg_rating) || 0;
            const savings = parseFloat(product.savings) || 0;
            const savingsPct = Math.round((parseFloat(product.savings_pct) || 0) * 100);
            const safeTitle = escapeHtml(product.title).replace(/'/g, "\\'");
            html += `
                <div class="chat-product-card" onclick="onProductClick(${product.product_id}, 'chatbot')">
                    <div class="cp-title">${escapeHtml(product.title)}</div>
                    <div class="cp-meta">
                        <span>$${price.toFixed(2)}</span>
                        <span>⭐ ${rating.toFixed(1)}</span>
                        <span>${escapeHtml(product.category)}</span>
                    </div>
                    ${savings > 0 ? `<div style="font-size:0.7rem; color:var(--amazon-green); font-weight:600; margin-top:3px;">Save $${savings.toFixed(2)} (${savingsPct}% off)</div>` : ''}
                    ${product.reason ? `<div style="font-size:0.7rem; color:var(--amazon-orange); margin-top:3px;">💡 ${escapeHtml(product.reason)}</div>` : ''}
                    <button onclick="event.stopPropagation(); addToCart(${product.product_id}, '${safeTitle}')" style="margin-top:4px; font-size:0.7rem; padding:2px 8px; border:1px solid var(--border-light); border-radius:4px; background:white; cursor:pointer;">+ Add to cart</button>
                    <button onclick="event.stopPropagation(); watchDealFromChat(${product.product_id})" style="margin-top:4px; margin-left:4px; font-size:0.7rem; padding:2px 8px; border:1px solid var(--border-light); border-radius:4px; background:white; cursor:pointer;">Watch price</button>
                </div>`;
        });
        html += '</div>';
    }

    (pendingActions || []).forEach(action => {
        html += renderPendingAction(action);
    });

    messageEl.innerHTML = html;
    if (role === 'assistant' && feedbackMeta) {
        const fb = document.createElement('div');
        fb.className = 'chat-feedback';
        const up = document.createElement('button');
        up.type = 'button';
        up.title = 'Helpful';
        up.textContent = '👍';
        const down = document.createElement('button');
        down.type = 'button';
        down.title = 'Not helpful';
        down.textContent = '👎';
        const meta = {
            message: feedbackMeta.message || '',
            response: text || '',
            tools_used: feedbackMeta.tools_used || [],
            thread_id: feedbackMeta.thread_id || null,
        };
        up.onclick = () => sendChatFeedback('up', meta, fb);
        down.onclick = () => sendChatFeedback('down', meta, fb);
        fb.appendChild(up);
        fb.appendChild(down);
        messageEl.appendChild(fb);
    }
    container.appendChild(messageEl);

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * A cart change proposed by the agent. Nothing changes until the member confirms.
 */
function renderPendingAction(action) {
    const verbs = { add: `Add ${action.quantity} ×`, set: `Set quantity to ${action.quantity} for`, remove: 'Remove' };
    const title = escapeHtml(action.product.title);
    const price = parseFloat(action.product.price) || 0;
    return `
        <div id="cart-action-${action.action_id}" class="pending-cart-action">
            <div class="pending-cart-title">${escapeHtml(verbs[action.action] || action.action)} ${title}</div>
            <div class="pending-cart-meta">$${price.toFixed(2)} each${action.reason ? ' · ' + escapeHtml(action.reason) : ''}</div>
            <div class="cart-action-buttons">
                <button type="button" class="pending-confirm" onclick="resolveCartAction('${action.action_id}', 'confirm')">Confirm</button>
                <button type="button" class="pending-dismiss" onclick="resolveCartAction('${action.action_id}', 'reject')">Dismiss</button>
            </div>
        </div>`;
}

async function resolveCartAction(actionId, decision) {
    const card = document.getElementById(`cart-action-${actionId}`);
    const buttons = card.querySelector('.cart-action-buttons');
    buttons.querySelectorAll('button').forEach(b => b.disabled = true);

    try {
        const response = await fetch(`/api/cart/actions/${actionId}/${decision}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: cartUserId() }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Server error: ${response.status}`);

        if (decision === 'confirm') {
            applyCartUpdate(data);
            showToast('Cart updated');
        }
        buttons.innerHTML = `<span style="font-size:0.7rem; color:var(--text-light);">${decision === 'confirm' ? '✓ Cart updated' : 'Dismissed'}</span>`;
    } catch (error) {
        console.error('Cart action error:', error);
        showToast(error.message || 'Could not update your cart.', 'error');
        buttons.querySelectorAll('button').forEach(b => b.disabled = false);
    }
}

/**
 * Add typing indicator ("..." animation).
 */
function addTypingIndicator() {
    const container = document.getElementById('chatMessages');
    const id = 'typing-' + Date.now();

    const el = document.createElement('div');
    el.className = 'chat-message assistant';
    el.id = id;
    el.innerHTML = `
        <span style="display:inline-flex;gap:4px;align-items:center;">
            <span style="animation:pulse 1s infinite;">●</span>
            <span style="animation:pulse 1s 0.2s infinite;">●</span>
            <span style="animation:pulse 1s 0.4s infinite;">●</span>
        </span>`;

    container.appendChild(el);
    container.scrollTop = container.scrollHeight;

    return id;
}

/**
 * Remove typing indicator.
 */
function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ========== Voice Recording ==========
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function toggleVoiceRecording() {
    const btn = document.getElementById('chatVoiceBtn');
    
    if (isRecording) {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        btn.classList.remove('recording');
        btn.textContent = '🎤';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            stream.getTracks().forEach(track => track.stop()); // release mic
            
            await sendAudioChat(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        btn.classList.add('recording');
        btn.textContent = '⏹️';

    } catch (err) {
        console.error("Error accessing microphone:", err);
        showToast("Could not access microphone.", "error");
    }
}

async function sendChatFeedback(rating, meta, el) {
    try {
        await fetch('/api/chat/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.currentUser || 1,
                rating,
                message: meta.message || '',
                response: meta.response || '',
                tools_used: meta.tools_used || [],
                thread_id: meta.thread_id || null,
            }),
        });
        if (el) el.innerHTML = `<span class="chat-feedback-done">${rating === 'up' ? 'Thanks!' : 'Got it'}</span>`;
    } catch (e) {
        console.error(e);
    }
}

async function watchDealFromChat(productId) {
    try {
        const res = await fetch('/api/deals/watch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.currentUser || 1, product_id: productId }),
        });
        if (!res.ok) throw new Error('watch failed');
        showToast('Watching for price drops');
    } catch (e) {
        showToast('Could not set price watch.', 'error');
    }
}

async function refreshDealAlerts() {
    const banner = document.getElementById('dealAlertsBanner');
    if (!banner) return;
    try {
        const data = await api(`/deals/alerts/${state.currentUser || 1}?limit=5`);
        const unread = (data.alerts || []).filter(a => !a.read);
        if (!unread.length) {
            banner.hidden = true;
            banner.innerHTML = '';
            return;
        }
        banner.hidden = false;
        banner.innerHTML = unread.slice(0, 2).map(a =>
            `<button type="button" onclick="sendSuggestion(${JSON.stringify(a.message || 'Show deal alerts')})">${escapeHtml(a.message || 'Price drop')}</button>`
        ).join('');
    } catch (_) { /* ignore */ }
}

async function sendImageChat(input) {
    const file = input.files && input.files[0];
    input.value = '';
    if (!file || isChatSending) return;

    const userId = state.currentUser || 1;
    addChatMessage('📷 Looking at your photo', 'user');
    const typingId = addTypingIndicator();
    isChatSending = true;
    document.getElementById('chatSendBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('user_id', userId);
        formData.append('message', 'Find a cheaper match for this product');
        formData.append('file', file, file.name || 'photo.jpg');

        const response = await fetch('/api/chat/image', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        removeTypingIndicator(typingId);
        if (data.transcription) {
            rememberChatTurn(userId, data.transcription, null);
        }
        if (state.currentUser === userId) {
            addChatMessage(data.response, 'assistant', data.products, data.pending_actions, {
                message: data.transcription || 'image cheaper match',
                tools_used: data.tools_used,
                thread_id: data.thread_id,
            });
            if (state.currentUser === userId && data.active_context && data.active_context.length) updateActiveContext(data.active_context);
        if (typeof applyUiActions === 'function') applyUiActions(data.ui_actions);
        }
        rememberChatTurn(userId, null, data.response);
    } catch (error) {
        console.error('Image chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage("Sorry, I couldn't process that photo.", 'assistant');
    } finally {
        isChatSending = false;
        document.getElementById('chatSendBtn').disabled = false;
    }
}

async function sendAudioChat(audioBlob) {
    if (isChatSending) return;

    const userId = state.currentUser || 1;
    const typingId = addTypingIndicator();
    isChatSending = true;
    document.getElementById('chatSendBtn').disabled = true;
    document.getElementById('chatVoiceBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('user_id', userId);
        formData.append('file', audioBlob, 'recording.webm');
        
        const apiHistory = chatHistory.slice(-6).map(msg => ({
            role: msg.role,
            content: msg.content,
        }));
        formData.append('conversation_history', JSON.stringify(apiHistory));

        const response = await fetch('/api/chat/audio', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        removeTypingIndicator(typingId);
        rememberChatTurn(userId, data.transcription || '', data.response);
        if (state.currentUser !== userId) return;

        if (data.transcription) addChatMessage(data.transcription, 'user');
        addChatMessage(data.response, 'assistant', data.products, data.pending_actions);
        if (typeof applyUiActions === 'function') applyUiActions(data.ui_actions);

    } catch (error) {
        console.error('Audio chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage('Sorry, I couldn\'t process your voice message.', 'assistant');
    } finally {
        isChatSending = false;
        document.getElementById('chatSendBtn').disabled = false;
        document.getElementById('chatVoiceBtn').disabled = false;
    }
}
