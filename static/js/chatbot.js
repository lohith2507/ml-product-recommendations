/**
 * AI Chatbot Widget
 * Handles the floating chat panel, message sending, and product rendering in chat.
 */

let chatHistory = [];
let isChatSending = false;

/**
 * Toggle the chatbot panel open/closed.
 */
function toggleChat() {
    const panel = document.getElementById('chatPanel');
    const toggle = document.getElementById('chatToggle');

    panel.classList.toggle('open');
    toggle.textContent = panel.classList.contains('open') ? '✕' : '💬';
}

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
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Add assistant response
        addChatMessage(data.response, 'assistant', data.products, data.pending_actions);

        // Update conversation history
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.response });

        // Update context tags
        if (data.active_context && data.active_context.length > 0) {
            updateActiveContext(data.active_context);
        }

    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage(
            'Sorry, I\'m having trouble connecting. Please try again!',
            'assistant'
        );
    } finally {
        isChatSending = false;
        document.getElementById('chatSendBtn').disabled = false;
        input.focus();
    }
}

function updateActiveContext(tags) {
    const contextContainer = document.getElementById('chatActiveContext');
    const tagsContainer = document.getElementById('chatContextTags');
    
    if (tags.length === 0) {
        contextContainer.style.display = 'none';
        return;
    }
    
    contextContainer.style.display = 'flex';
    tagsContainer.innerHTML = tags.map(tag => 
        `<span style="background:var(--amazon-blue); color:white; padding:2px 6px; border-radius:4px; font-size:0.7rem;">${escapeHtml(tag)}</span>`
    ).join('');
}

function clearChatMemory() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = `
        <div class="chat-message assistant">
            Hi! I've cleared my memory. How can I help you today?
        </div>
    `;
    updateActiveContext([]);
}

/**
 * Add a message bubble to the chat panel.
 */
function addChatMessage(text, role, products = [], pendingActions = []) {
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
                </div>`;
        });
        html += '</div>';
    }

    (pendingActions || []).forEach(action => {
        html += renderPendingAction(action);
    });

    messageEl.innerHTML = html;
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
        <div id="cart-action-${action.action_id}" style="margin-top:8px; padding:8px; border:1px solid var(--amazon-orange); border-radius:6px; background:white;">
            <div style="font-size:0.75rem; font-weight:600;">${escapeHtml(verbs[action.action] || action.action)} ${title}</div>
            <div style="font-size:0.7rem; color:var(--text-light);">$${price.toFixed(2)} each${action.reason ? ' · ' + escapeHtml(action.reason) : ''}</div>
            <div class="cart-action-buttons" style="display:flex; gap:6px; margin-top:6px;">
                <button onclick="resolveCartAction('${action.action_id}', 'confirm')" style="font-size:0.7rem; padding:3px 10px; border:none; border-radius:4px; background:var(--amazon-orange); color:white; cursor:pointer;">Confirm</button>
                <button onclick="resolveCartAction('${action.action_id}', 'reject')" style="font-size:0.7rem; padding:3px 10px; border:1px solid var(--border-light); border-radius:4px; background:white; cursor:pointer;">Dismiss</button>
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
        
        // Show what was transcribed
        if (data.transcription) {
            addChatMessage(data.transcription, 'user');
            chatHistory.push({ role: 'user', content: data.transcription });
        }

        addChatMessage(data.response, 'assistant', data.products, data.pending_actions);
        chatHistory.push({ role: 'assistant', content: data.response });

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
