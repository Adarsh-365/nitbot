document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window");
    const chatStage = document.querySelector(".chat-stage");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const composerWrap = document.querySelector(".composer-wrap");
    const composerSpacer = document.getElementById("composer-spacer");

    let shouldStickToBottom = true;

    function syncComposerOffset() {
        if (!chatStage || !composerWrap || !composerSpacer) {
            return;
        }
        const composerHeight = Math.ceil(composerWrap.getBoundingClientRect().height);
        const bottomOffset = composerHeight + 16;
        chatStage.style.paddingBottom = `${bottomOffset}px`;
        composerSpacer.style.bottom = `${bottomOffset}px`;
    }

    function scrollToBottom() {
        if (!chatStage) {
            return;
        }
        chatStage.scrollTop = chatStage.scrollHeight;
        shouldStickToBottom = true;
    }

    function handleChatScroll() {
        if (!chatStage) {
            return;
        }
        const distanceFromBottom =
            chatStage.scrollHeight - chatStage.scrollTop - chatStage.clientHeight;
        shouldStickToBottom = distanceFromBottom < 80;
    }

    function maybeScrollToBottom() {
        if (shouldStickToBottom) {
            scrollToBottom();
        }
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    function stripMarkdownForPreview(text) {
        return text
            .replace(/```[\s\S]*?```/g, (match) => match.replace(/```/g, "").trim())
            .replace(/^#{1,6}\s+/gm, "")
            .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, "$1")
            .replace(/\*\*([^*]+)\*\*/g, "$1")
            .replace(/\*([^*]+)\*/g, "$1")
            .replace(/`([^`]+)`/g, "$1")
            .replace(/^>\s?/gm, "")
            .replace(/^\d+\.\s+/gm, "")
            .replace(/^[-*]\s+/gm, "")
            .replace(/\r\n/g, "\n");
    }

    function renderPreviewText(text) {
        const safe = escapeHtml(text);
        return safe
            .split("\n\n")
            .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
            .join("");
    }

    function renderMarkdown(text) {
        if (window.marked && typeof window.marked.parse === "function") {
            return window.marked.parse(text, {
                gfm: true,
                breaks: true,
            });
        }
        return renderPreviewText(text);
    }

    function createMessage(role, content, metaText) {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;

        const meta = document.createElement("div");
        meta.className = "message-meta";
        meta.textContent = metaText;

        const bubble = document.createElement("div");
        bubble.className = `message-bubble ${role === "user" ? "message-bubble-user" : "message-bubble-assistant"}`;

        const body = document.createElement("div");
        body.className = role === "user" ? "message-plain" : "message-markdown";

        if (role === "user") {
            const p = document.createElement("p");
            p.textContent = content;
            body.appendChild(p);
        } else {
            body.innerHTML = renderMarkdown(content);
        }

        bubble.appendChild(body);
        wrapper.appendChild(meta);
        wrapper.appendChild(bubble);
        return wrapper;
    }

    function showTypingIndicator() {
        const wrapper = document.createElement("div");
        wrapper.className = "message assistant-message";
        wrapper.id = "typing-indicator";

        const meta = document.createElement("div");
        meta.className = "message-meta";
        meta.textContent = "Node_Response_Beta // Calculating";

        const row = document.createElement("div");
        row.className = "typing-row";

        const label = document.createElement("div");
        label.className = "typing-label";
        label.textContent = "Thinking";

        const dots = document.createElement("div");
        dots.className = "typing-dots";
        dots.innerHTML = "<span></span><span></span><span></span>";

        row.appendChild(label);
        row.appendChild(dots);
        wrapper.appendChild(meta);
        wrapper.appendChild(row);
        chatWindow.appendChild(wrapper);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) {
            indicator.remove();
        }
    }

    async function typeAssistantMessage(text) {
        const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const wrapper = createMessage("assistant", "", `Node_Response_Beta // ${timestamp}`);
        const body = wrapper.querySelector(".message-markdown");
        chatWindow.appendChild(wrapper);
        maybeScrollToBottom();

        if (document.hidden) {
            body.innerHTML = renderMarkdown(text);
            scrollToBottom();
            return;
        }

        const previewText = stripMarkdownForPreview(text);
        const characters = Array.from(previewText);
        let current = "";
        for (let index = 0; index < characters.length; index += 1) {
            current += characters[index];
            body.innerHTML = renderPreviewText(current);
            maybeScrollToBottom();
            await new Promise((resolve) => setTimeout(resolve, 8));
        }

        body.innerHTML = renderMarkdown(text);
        maybeScrollToBottom();
    }

    function addUserMessage(text) {
        const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        chatWindow.appendChild(createMessage("user", text, `User_Origin // ${timestamp}`));
        scrollToBottom();
    }

    function autoResize() {
        messageInput.style.height = "auto";
        messageInput.style.height = `${Math.min(messageInput.scrollHeight, 160)}px`;
        messageInput.style.overflowY = messageInput.scrollHeight > 160 ? "auto" : "hidden";
        syncComposerOffset();
        maybeScrollToBottom();
    }

    async function handleSendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) {
            return;
        }

        addUserMessage(messageText);
        messageInput.value = "";
        autoResize();
        messageInput.focus();
        sendButton.disabled = true;
        showTypingIndicator();

        try {
            const response = await fetch("/callbot/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ userMessage: messageText }),
            });

            const data = await response.json();
            hideTypingIndicator();
            await typeAssistantMessage(data.botText || "I could not find a response.");
        } catch (_error) {
            hideTypingIndicator();
            chatWindow.appendChild(
                createMessage(
                    "assistant",
                    "Sorry, there was an error while contacting the backend. Please try again.",
                    "Node_Response_Beta // Error"
                )
            );
            scrollToBottom();
        } finally {
            sendButton.disabled = false;
            if (!document.hidden) {
                messageInput.focus();
            }
        }
    }

    sendButton.addEventListener("click", handleSendMessage);

    messageInput.addEventListener("input", autoResize);
    if (chatStage) {
        chatStage.addEventListener("scroll", handleChatScroll);
    }

    if (typeof ResizeObserver !== "undefined" && composerWrap) {
        const resizeObserver = new ResizeObserver(() => {
            syncComposerOffset();
            maybeScrollToBottom();
        });
        resizeObserver.observe(composerWrap);
    }

    messageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    });

    syncComposerOffset();
    autoResize();
    scrollToBottom();
});
