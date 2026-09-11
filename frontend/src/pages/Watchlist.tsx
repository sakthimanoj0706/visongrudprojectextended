import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, TextField, Button, Grid,
  Select, MenuItem, InputLabel, FormControl, Alert, CircularProgress,
  Paper
} from '@mui/material';
import { CloudUpload as CloudUploadIcon, PersonAdd as PersonAddIcon } from '@mui/icons-material';
import apiClient from '../api/client';

export const Watchlist: React.FC = () => {
  const [name, setName] = useState('');
  const [category, setCategory] = useState('Watchlist');
  const [riskLevel, setRiskLevel] = useState('Medium');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    }
  };

  const handleEnroll = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !file) {
      setStatusMsg({ type: 'error', text: 'Name and Image are required' });
      return;
    }

    setLoading(true);
    setStatusMsg(null);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('category', category);
    formData.append('risk_level', riskLevel);
    formData.append('image', file);

    try {
      const res = await apiClient.post('/api/v1/watchlist/enroll', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatusMsg({ type: 'success', text: `Successfully enrolled ${res.data.name} (ID: ${res.data.person_id})` });
      
      // Reset form on success
      setName('');
      setCategory('Watchlist');
      setRiskLevel('Medium');
      setFile(null);
      setPreviewUrl(null);
    } catch (err: any) {
      console.error(err);
      setStatusMsg({ type: 'error', text: err.response?.data?.detail || 'Enrollment failed' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h5" sx={{ mb: 4, fontWeight: 700 }}>
        Watchlist Registry
      </Typography>

      <Grid container spacing={4}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                <PersonAddIcon color="primary" /> Enroll Target Identity
              </Typography>

              {statusMsg && (
                <Alert severity={statusMsg.type} sx={{ mb: 3 }}>
                  {statusMsg.text}
                </Alert>
              )}

              <Box component="form" onSubmit={handleEnroll} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <TextField
                  label="Target Name"
                  variant="outlined"
                  fullWidth
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />

                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={category}
                    label="Category"
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    <MenuItem value="Watchlist">Watchlist</MenuItem>
                    <MenuItem value="VIP">VIP</MenuItem>
                    <MenuItem value="Employee">Employee</MenuItem>
                    <MenuItem value="Unknown">Unknown</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Risk Level</InputLabel>
                  <Select
                    value={riskLevel}
                    label="Risk Level"
                    onChange={(e) => setRiskLevel(e.target.value)}
                  >
                    <MenuItem value="Low">Low</MenuItem>
                    <MenuItem value="Medium">Medium</MenuItem>
                    <MenuItem value="High">High</MenuItem>
                    <MenuItem value="Critical">Critical</MenuItem>
                  </Select>
                </FormControl>

                <Box>
                  <input
                    accept="image/*"
                    style={{ display: 'none' }}
                    id="raised-button-file"
                    type="file"
                    onChange={handleFileChange}
                  />
                  <label htmlFor="raised-button-file">
                    <Button variant="outlined" component="span" startIcon={<CloudUploadIcon />} fullWidth>
                      Upload Target Photo
                    </Button>
                  </label>
                </Box>

                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={loading || !file || !name}
                  sx={{ mt: 2 }}
                >
                  {loading ? <CircularProgress size={24} color="inherit" /> : 'Enroll Target'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 6 }}>
          {previewUrl && (
            <Card>
              <CardContent>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Image Preview
                </Typography>
                <Paper sx={{ p: 1, border: '1px solid rgba(255,255,255,0.1)', bgcolor: '#000', display: 'flex', justifyContent: 'center' }}>
                  <img src={previewUrl} alt="Preview" style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }} />
                </Paper>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Watchlist;
