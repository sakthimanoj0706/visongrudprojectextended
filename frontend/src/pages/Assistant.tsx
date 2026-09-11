import React, { useState, useRef, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, TextField, IconButton,
  Paper, CircularProgress, Chip, Divider, Avatar
} from '@mui/material';
import { Send as SendIcon, SmartToy as BotIcon, Person as PersonIcon } from '@mui/icons-material';
import apiClient from '../api/client';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
}

export const Assistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello! I am the VisionGuard RAG Assistant. How can I help you investigate today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    
    // Add user message to UI immediately
    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setLoading(true);

    try {
      // Format history for backend
      const history = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content }));

      const res = await apiClient.post('/api/v1/surveillance/assistant/chat', {
        message: userMessage,
        history: history
      });

      const assistantMsg = res.data.response || "No response received.";
      const sources = res.data.sources || [];

      setMessages(prev => [...prev, { role: 'assistant', content: assistantMsg, sources }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: Could not connect to RAG Assistant.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
        NL Investigation Assistant
      </Typography>

      <Card sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)' }}>
        {/* Chat History Area */}
        <CardContent sx={{ flexGrow: 1, overflowY: 'auto', p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
          {messages.map((msg, idx) => (
            <Box key={idx} sx={{ display: 'flex', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', gap: 2 }}>
              <Avatar sx={{ bgcolor: msg.role === 'user' ? 'primary.main' : 'secondary.main' }}>
                {msg.role === 'user' ? <PersonIcon /> : <BotIcon />}
              </Avatar>
              <Paper sx={{ 
                p: 2, 
                maxWidth: '75%', 
                bgcolor: msg.role === 'user' ? 'primary.dark' : 'rgba(5, 7, 13, 0.6)',
                borderRadius: 2,
                border: msg.role === 'assistant' ? '1px solid rgba(255,255,255,0.1)' : 'none'
              }}>
                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </Typography>
                
                {msg.sources && msg.sources.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ mb: 1, borderColor: 'rgba(255,255,255,0.1)' }} />
                    <Typography variant="caption" color="text.secondary">Sources Context:</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {msg.sources.map((src, i) => (
                        <Chip 
                          key={i} 
                          label={`${src.entity_type.toUpperCase()} | ${new Date(src.timestamp).toLocaleTimeString()}`} 
                          size="small" 
                          variant="outlined" 
                          color="info"
                          sx={{ fontSize: '10px' }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>
            </Box>
          ))}
          {loading && (
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Avatar sx={{ bgcolor: 'secondary.main' }}><BotIcon /></Avatar>
              <Paper sx={{ p: 2, bgcolor: 'rgba(5, 7, 13, 0.6)', display: 'flex', alignItems: 'center' }}>
                <CircularProgress size={20} />
              </Paper>
            </Box>
          )}
          <div ref={messagesEndRef} />
        </CardContent>

        {/* Input Area */}
        <Divider />
        <Box component="form" onSubmit={handleSend} sx={{ p: 2, bgcolor: 'background.paper', display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Ask about recent events, targets, or anomalies..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoComplete="off"
          />
          <IconButton type="submit" color="primary" disabled={loading || !input.trim()} sx={{ p: 2, bgcolor: 'rgba(0, 168, 255, 0.1)' }}>
            <SendIcon />
          </IconButton>
        </Box>
      </Card>
    </Box>
  );
};

export default Assistant;
