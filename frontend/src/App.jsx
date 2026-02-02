import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AppBar,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  InputLabel,
  Link,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Toolbar,
  Typography
} from "@mui/material";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const brandOptions = ["", "beko", "electroline", "hisense"];

function MessageBubble({ role, text }) {
  const isUser = role === "user";
  return (
    <Box
      sx={{
        alignSelf: isUser ? "flex-end" : "flex-start",
        bgcolor: isUser ? "primary.main" : "grey.200",
        color: isUser ? "primary.contrastText" : "text.primary",
        px: 2,
        py: 1.5,
        borderRadius: 2,
        maxWidth: "80%"
      }}
    >
      {isUser ? (
        <Typography variant="body1" whiteSpace="pre-wrap">
          {text}
        </Typography>
      ) : (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <Typography variant="body1" paragraph>{children}</Typography>,
            ul: ({ children }) => <Typography component="ul" sx={{ my: 1 }}>{children}</Typography>,
            ol: ({ children }) => <Typography component="ol" sx={{ my: 1 }}>{children}</Typography>,
            li: ({ children }) => <Typography component="li" variant="body2">{children}</Typography>,
            code: ({ inline, children }) =>
              inline ? (
                <Typography component="code" sx={{ bgcolor: "grey.300", px: 0.5, borderRadius: 0.5, fontFamily: "monospace" }}>
                  {children}
                </Typography>
              ) : (
                <Typography component="pre" sx={{ bgcolor: "grey.800", color: "white", p: 1.5, borderRadius: 1, overflow: "auto", fontFamily: "monospace" }}>
                  <code>{children}</code>
                </Typography>
              ),
            a: ({ href, children }) => {
              const fullUrl = href?.startsWith('/') ? `${API_BASE}${href}` : href;
              const isImage = href?.startsWith('/static/images/') || href?.match(/\.(png|jpe?g|gif|webp)$/i);
              
              if (isImage) {
                return (
                  <Box sx={{ my: 2 }}>
                    <img src={fullUrl} alt={children} style={{ maxWidth: '100%', borderRadius: 8 }} />
                    <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
                      {children}
                    </Typography>
                  </Box>
                );
              }
              
              return <Link href={fullUrl} target="_blank" rel="noopener noreferrer">{children}</Link>;
            },
            h1: ({ children }) => <Typography variant="h5" gutterBottom>{children}</Typography>,
            h2: ({ children }) => <Typography variant="h6" gutterBottom>{children}</Typography>,
            h3: ({ children }) => <Typography variant="subtitle1" gutterBottom fontWeight="bold">{children}</Typography>,
          }}
        >
          {text}
        </ReactMarkdown>
      )}
    </Box>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Ciao! Posso aiutarti con i manuali. Scrivi una domanda oppure avvia l'ingestione."
    }
  ]);
  const [input, setInput] = useState("");
  const [brand, setBrand] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const [images, setImages] = useState([]);
  const [answerText, setAnswerText] = useState("");
  const [ingestStatus, setIngestStatus] = useState(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const handleSend = async () => {
    if (!canSend) return;
    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, brand: brand || null, top_k: 5 })
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", text: data.answer }]);
      setAnswerText(data.answer || "");
      setSources(data.sources || []);
      setImages(data.images || []);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Errore nella chiamata API." }
      ]);
      setAnswerText("");
    } finally {
      setLoading(false);
    }
  };

  const handleIngest = async () => {
    setIngestStatus("loading");
    try {
      const response = await fetch(`${API_BASE}/ingest`, { method: "POST" });
      const data = await response.json();
      setIngestStatus(`Indicizzati ${data.manuals} manuali, ${data.chunks} chunk, ${data.images} immagini.`);
    } catch (error) {
      setIngestStatus("Errore durante ingestione.");
    }
  };

  const citedImages = useMemo(() => {
    return images;
  }, [images]);

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "grey.100" }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Customer Assistant</Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel id="brand-label">Marca</InputLabel>
              <Select
                labelId="brand-label"
                label="Marca"
                value={brand}
                onChange={(event) => setBrand(event.target.value)}
              >
                {brandOptions.map((item) => (
                  <MenuItem key={item} value={item}>
                    {item === "" ? "Tutte" : item}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button variant="contained" onClick={handleIngest} disabled={ingestStatus === "loading"}>
              Indicizza manuali
            </Button>
            {ingestStatus && (
              <Chip
                color={ingestStatus === "loading" ? "default" : "success"}
                label={ingestStatus === "loading" ? "Ingestione in corso..." : ingestStatus}
              />
            )}
          </Stack>
        </Paper>

        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Stack spacing={2} sx={{ minHeight: 360 }}>
            {messages.map((msg, index) => (
              <MessageBubble key={`${msg.role}-${index}`} role={msg.role} text={msg.text} />
            ))}
            {loading && (
              <Box sx={{ display: "flex", justifyContent: "center" }}>
                <CircularProgress size={24} />
              </Box>
            )}
          </Stack>

          <Divider sx={{ my: 2 }} />

          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <TextField
              fullWidth
              placeholder="Scrivi la tua domanda..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleSend();
                }
              }}
            />
            <Button variant="contained" onClick={handleSend} disabled={!canSend}>
              Invia
            </Button>
          </Stack>
        </Paper>

        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Fonti
          </Typography>
          {sources.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              Nessuna fonte disponibile.
            </Typography>
          ) : (
            <Stack spacing={1}>
              {sources.map((source) => (
                <Box key={source.chunk_id}>
                  <Link href={`${API_BASE}${source.link}`} target="_blank" rel="noopener">
                    {source.manual} - Pagina {source.page}
                  </Link>
                  <Typography variant="caption" display="block" color="text.secondary">
                    {source.brand} · score {source.score.toFixed(3)}
                  </Typography>
                </Box>
              ))}
            </Stack>
          )}

          <Divider sx={{ my: 2 }} />

          <Typography variant="h6" sx={{ mb: 2 }}>
            Immagini correlate
          </Typography>
          {citedImages.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              Nessuna immagine disponibile.
            </Typography>
          ) : (
            <Stack direction="row" spacing={2} flexWrap="wrap">
              {citedImages.map((img) => (
                <Box key={`${img.chunk_id}-${img.url}`} sx={{ width: 160 }}>
                  <img
                    src={`${API_BASE}${img.url}`}
                    alt="manuale"
                    style={{ width: "100%", borderRadius: 8 }}
                  />
                </Box>
              ))}
            </Stack>
          )}
        </Paper>
      </Container>
    </Box>
  );
}
