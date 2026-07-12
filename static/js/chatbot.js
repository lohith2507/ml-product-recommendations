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
        addChatMessage(data.response, 'assistant', data.products);

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
function addChatMessage(text, role, products = []) {
    const container = document.getElementById('chatMessages');

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${role}`;

    let html = escapeHtml(text);

    // Add product cards if present
    if (products && products.length > 0) {
        html += '<div class="chat-products">';
        products.forEach(product => {
            const price = parseFloat(product.price) || 0;
            const rating = parseFloat(product.avg_rating) || 0;
            html += `
                <div class="chat-product-card" onclick="onProductClick(${product.product_id}, 'chatbot')">
                    <div class="cp-title">${escapeHtml(product.title)}</div>
                    <div class="cp-meta">
                        <span>$${price.toFixed(2)}</span>
                        <span>⭐ ${rating.toFixed(1)}</span>
                        <span>${escapeHtml(product.category)}</span>
                    </div>
                    ${product.reason ? `<div style="font-size:0.7rem; color:var(--amazon-orange); margin-top:3px;">💡 ${escapeHtml(product.reason)}</div>` : ''}
                </div>`;
        });
        html += '</div>';
    }

    messageEl.innerHTML = html;
    container.appendChild(messageEl);

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
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

        addChatMessage(data.response, 'assistant', data.products);
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
