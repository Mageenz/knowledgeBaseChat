<template>
  <div class="chat-container">
    <div class="chat-messages" ref="messageContainer">
      <div v-for="(message, index) in messages" :key="index" 
           :class="['message', message.type === 'user' ? 'user-message' : 'bot-message']">
        <div class="message-content">{{ message.content }}</div>
      </div>
    </div>
    <div class="chat-input">
      <input 
        v-model="userInput" 
        @keyup.enter="sendMessage"
        placeholder="请输入您的问题..."
        type="text"
      >
      <button @click="sendMessage" :disabled="thinking">发送</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatComponent',
  data() {
    return {
      messages: [],
      userInput: '',
      thinking: false
    }
  },
  methods: {
    async sendMessage() {
      if (!this.userInput.trim() || this.thinking) return;
      
      // 添加用户消息
      this.messages.push({
        type: 'user',
        content: this.userInput
      });
      
      const userQuestion = this.userInput;
      this.userInput = '';
      
      // 添加一个空的机器人消息，并设置pending状态
      this.messages.push({
        type: 'bot',
        content: '',
        thinking: true
      });
      
      this.thinking = true;
      
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: [
              {"role": "user", "content": userQuestion}
            ]
          })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              // 更新最后一条机器人消息的内容
              const lastMessage = this.messages[this.messages.length - 1];
              if (lastMessage.type === 'bot') {
                lastMessage.content += data.content;
                this.$nextTick(() => {
                  this.scrollToBottom();
                });
              }
            }
          }
        }
      } catch (error) {
        console.error('发送消息失败:', error);
        const lastMessage = this.messages[this.messages.length - 1];
        if (lastMessage.type === 'bot') {
          lastMessage.content = '抱歉，发生了一些错误，请稍后重试。';
        }
      } finally {
        // 移除pending状态
        const lastMessage = this.messages[this.messages.length - 1];
        if (lastMessage.type === 'bot') {
          lastMessage.thinking = false;
        }
        this.thinking = false;
      }
    },
    scrollToBottom() {
      const container = this.$refs.messageContainer;
      container.scrollTop = container.scrollHeight;
    }
  }
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  box-sizing: border-box;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
  margin-bottom: 20px;
  height: calc(100vh - 120px); /* 减去输入框和padding的高度 */
  min-height: 0; /* 确保flex子元素可以正确滚动 */
}

.message {
  margin-bottom: 15px;
  max-width: 80%;
}

.user-message {
  margin-left: auto;
}

.bot-message {
  margin-right: auto;
}

.message-content {
  padding: 10px 15px;
  border-radius: 15px;
  display: inline-block;
}

.user-message .message-content {
  background: #007AFF;
  color: white;
}

.bot-message .message-content {
  background: white;
  color: #333;
}

.chat-input {
  display: flex;
  gap: 10px;
  padding: 10px 0;
  background: white;
  position: sticky;
  bottom: 0;
  z-index: 1;
}

.chat-input input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
}

.chat-input button {
  padding: 10px 20px;
  background: #007AFF;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

.chat-input button:hover {
  background: #0056b3;
}
</style>
