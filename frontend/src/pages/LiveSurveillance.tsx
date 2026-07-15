import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  TextField,
  CircularProgress,
  Grid
} from '@mui/material';
import { PlayArrow, Stop, Videocam } from '@mui/icons-material';
import apiClient from '../api/client';

interface ActiveStream {
  camera_id: string;
  location: string;
  stream_source: string;
  threshold: number;
}

export const LiveSurveillance: React.FC = () => {
  const [activeStreams, setActiveStreams] = useState<ActiveStream[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Form state
  const [cameraId, setCameraId] = useState('WEBCAM_0');
  const [location, setLocation] = useState('Operator Office');
  const [streamSource, setStreamSource] = useState('0');
  
  const fetchStreams = async () => {
    try {
      const res = await apiClient.get('/api/v1/surveillance/streams');
      setActiveStreams(res.data);
    } catch (e) {
      console.error("Failed to fetch running streams:", e);
    }
  };

  useEffect(() => {
    fetchStreams();
    const interval = setInterval(fetchStreams, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStartStream = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('camera_id', cameraId);
      formData.append('location', location);
      formData.append('stream_source', streamSource);
      formData.append('threshold', '0.40');

      await apiClient.post('/api/v1/surveillance/streams/start', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      fetchStreams();
    } catch (err) {
      console.error("Failed to start stream:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleStopStream = async (id: string) => {
    try {
      await apiClient.post(`/api/v1/surveillance/streams/stop/${id}`);
      fetchStreams();
    } catch (err) {
      console.error("Failed to stop stream:", err);
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={3}>
        {/* Controls Card */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                Stream Control Panel
              </Typography>
              <Box component="form" onSubmit={handleStartStream} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Camera ID"
                  value={cameraId}
                  onChange={(e) => setCameraId(e.target.value)}
                  required
                  size="small"
                />
                <TextField
                  label="Location"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  required
                  size="small"
                />
                <TextField
                  label="Source (RTSP/File/Device Index)"
                  value={streamSource}
                  onChange={(e) => setStreamSource(e.target.value)}
                  helperText="Use '0' for local webcam"
                  required
                  size="small"
                />
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={loading}
                  startIcon={<PlayArrow />}
                  sx={{ py: 1 }}
                >
                  {loading ? <CircularProgress size={24} /> : "Start Stream Feed"}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Video Feeds Grid */}
        <Grid size={{ xs: 12, md: 8 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 700 }}>
                Live Monitoring Streams
              </Typography>
              
              <Grid container spacing={3}>
                {activeStreams.map((stream) => (
                  <Grid key={stream.camera_id} size={{ xs: 12 }}>
                    <Card sx={{ bgcolor: 'rgba(5, 7, 13, 0.4)', border: '1px solid rgba(26, 37, 60, 0.4)' }}>
                      <CardContent sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Videocam color="primary" />
                            <Typography variant="body1" sx={{ fontWeight: 700 }}>
                              {stream.camera_id} ({stream.location})
                            </Typography>
                          </Box>
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            startIcon={<Stop />}
                            onClick={() => handleStopStream(stream.camera_id)}
                          >
                            Terminate
                          </Button>
                        </Box>
                        
                        {/* Live MJPEG Image Feed */}
                        <Box
                          sx={{
                            width: '100%',
                            height: 380,
                            borderRadius: '8px',
                            overflow: 'hidden',
                            border: '1px solid rgba(26, 37, 60, 0.2)',
                            bgcolor: 'black',
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            position: 'relative'
                          }}
                        >
                          <img
                            src={`/api/v1/surveillance/streams/video/${stream.camera_id}`}
                            alt={`Live Stream: ${stream.camera_id}`}
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}

                {activeStreams.length === 0 && (
                  <Grid size={{ xs: 12 }}>
                    <Box sx={{ py: 6, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        No active monitoring channels detected. Use the control panel to start a stream.
                      </Typography>
                    </Box>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveSurveillance;
