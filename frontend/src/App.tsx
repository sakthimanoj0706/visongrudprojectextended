import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { socTheme } from './theme/socTheme';
import AppRoutes from './routes/AppRoutes';

export const App: React.FC = () => {
  return (
    <ThemeProvider theme={socTheme}>
      <CssBaseline />
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
