import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./App.css";

const API_URL = "http://localhost:8000";

const cleanSpeciesName = (name) => {
  return name
    .replace("_sound", "")
    .replaceAll("_", " ")
    .replace("Darwins", "Darwin's")
    .trim();
};

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [visualMode, setVisualMode] = useState("3d");
  useEffect(() => {
  if (!file) {
    setAudioUrl("");
    return;
  }

  const url = URL.createObjectURL(file);
  setAudioUrl(url);

  return () => {
    URL.revokeObjectURL(url);
  };
}, [file]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data.prediction);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to BirdSense-AI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="background-glow glow-one"></div>
      <div className="background-glow glow-two"></div>

      <header className="navbar">
        <div className="logo">
          <span className="logo-icon">🐦</span>
          <span>BirdSense<span className="logo-accent">-AI</span></span>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI System Online
        </div>
      </header>

      <main className="main-content">

        <section className="hero">
          <div className="hero-badge">
            <span>✦</span>
            AI-POWERED BIRD IDENTIFICATION
          </div>

          <h1>
            Discover the bird
            <br />
            <span>behind the sound.</span>
          </h1>

          <p>
            Upload a bird recording and let BirdSense-AI
            analyze its acoustic signature using deep learning.
          </p>
        </section>

        <section className="analyzer-card">

          <div className="upload-area">
            <input
              id="audio-upload"
              type="file"
              accept=".wav,.mp3,.ogg,.flac,.m4a,audio/*"
              onChange={handleFileChange}
              hidden
            />

            <label
              htmlFor="audio-upload"
              className={`upload-box ${file ? "has-file" : ""}`}
            >
              <div className="upload-icon">
                {file ? "🎵" : "🎙️"}
              </div>

              {file ? (
                <>
                  <h3>{file.name}</h3>
                  <p>Audio file selected</p>
                </>
              ) : (
                <>
                  <h3>Drop your bird recording here</h3>
                  <p>or click to browse your device</p>
                  <span className="formats">
                    WAV • MP3 • OGG • FLAC • M4A
                  </span>
                </>
              )}
            </label>
            {file && audioUrl && (
  <audio
    controls
    preload="metadata"
    className="audio-player"
    src={audioUrl}
  />
)}
          </div>

          <button
            className="analyze-button"
            onClick={handlePredict}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing acoustic signature...
              </>
            ) : (
              <>
                Analyze Bird Sound
                <span>→</span>
              </>
            )}
          </button>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </section>

        {result && (
          <section className="results">

            <div className="result-header">
              <div>
                <span className="section-label">
                  IDENTIFICATION RESULT
                </span>

                <h2>{cleanSpeciesName(result.species)}</h2>
              </div>

              <div className="confidence">
                <span>CONFIDENCE</span>
                <strong>{result.confidence.toFixed(2)}%</strong>
              </div>
            </div>

            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${Math.min(result.confidence, 100)}%`,
                }}
              ></div>
            </div>

            <div className="top-predictions">
              <div className="section-title">
                <span>TOP PREDICTIONS</span>
                <span>MODEL OUTPUT</span>
              </div>

              {result.top_predictions.map((prediction, index) => (
                <div
                  className="prediction-row"
                  key={`${prediction.species}-${index}`}
                >
                  <div className="prediction-rank">
                    #{index + 1}
                  </div>

                  <div className="prediction-name">
                    {cleanSpeciesName(prediction.species)}
                  </div>

                  <div className="prediction-progress">
                    <div
                      style={{
                        width: `${Math.min(
                          prediction.confidence,
                          100
                        )}%`,
                      }}
                    ></div>
                  </div>

                  <div className="prediction-confidence">
                    {prediction.confidence.toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

                {result?.visualization && (
          <section className="acoustic-visualization">

            <div className="visualization-header">
              <div>
                <span className="section-label">
                  ACOUSTIC ANALYSIS
                </span>

                <h2>
                  Visualize the bird's acoustic signature
                </h2>

                <p>
                  Interactive frequency analysis generated from
                  the uploaded recording.
                </p>
              </div>

              <div className="visualization-toggle">
                <button
                  className={visualMode === "3d" ? "active" : ""}
                  onClick={() => setVisualMode("3d")}
                >
                  3D View
                </button>

                <button
                  className={visualMode === "2d" ? "active" : ""}
                  onClick={() => setVisualMode("2d")}
                >
                  Spectrogram
                </button>
              </div>
            </div>

            <div className="plot-container">
{visualMode === "3d" ? (
  <Plot
    data={[
      {
        type: "surface",
        z: result.visualization.spectrogram,
        x: result.visualization.time,
        y: result.visualization.frequency,
        colorscale: "Viridis",
        showscale: true,

        colorbar: {
          title: {
            text: "dB",
            font: {
              color: "#9db5a8",
            },
          },
          tickfont: {
            color: "#9db5a8",
          },
        },

        hovertemplate:
          "Time: %{x:.2f}s" +
          "<br>Frequency: %{y:.0f} Hz" +
          "<br>Intensity: %{z:.2f} dB" +
          "<extra></extra>",
      },
    ]}

    layout={{
      title: {
        text: "3D Acoustic Frequency Surface",
        font: {
          color: "#e8fff2",
          size: 18,
        },
      },

      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",

      scene: {
        bgcolor: "rgba(0,0,0,0)",

        xaxis: {
          title: {
            text: "Time (seconds)",
            font: {
              color: "#9db5a8",
              size: 13,
            },
          },
          color: "#9db5a8",
          gridcolor: "#1b3027",
          zerolinecolor: "#1b3027",
        },

        yaxis: {
          title: {
            text: "Frequency (Hz)",
            font: {
              color: "#9db5a8",
              size: 13,
            },
          },
          color: "#9db5a8",
          gridcolor: "#1b3027",
          zerolinecolor: "#1b3027",
        },

        zaxis: {
          title: {
            text: "Intensity (dB)",
            font: {
              color: "#9db5a8",
              size: 13,
            },
          },
          color: "#9db5a8",
          gridcolor: "#1b3027",
          zerolinecolor: "#1b3027",
        },

        camera: {
          eye: {
            x: 1.6,
            y: 1.6,
            z: 1.2,
          },
        },
      },

      margin: {
        l: 0,
        r: 0,
        t: 50,
        b: 0,
      },

      height: 600,

      font: {
        color: "#e8fff2",
      },
    }}

    config={{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: [
        "lasso2d",
        "select2d",
      ],
    }}

    style={{
      width: "100%",
    }}
  />
) : (
  <Plot
    data={[
      {
        type: "heatmap",
        z: result.visualization.spectrogram,
        x: result.visualization.time,
        y: result.visualization.frequency,
        colorscale: "Viridis",

        colorbar: {
          title: {
            text: "dB",
            font: {
              color: "#9db5a8",
            },
          },
          tickfont: {
            color: "#9db5a8",
          },
        },

        hovertemplate:
          "Time: %{x:.2f}s" +
          "<br>Frequency: %{y:.0f} Hz" +
          "<br>Intensity: %{z:.2f} dB" +
          "<extra></extra>",
      },
    ]}

    layout={{
      title: {
        text: "Bird Call Spectrogram",
        font: {
          color: "#e8fff2",
          size: 18,
        },
      },

      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",

      xaxis: {
        title: {
          text: "Time (seconds)",
          font: {
            color: "#9db5a8",
          },
        },
        color: "#9db5a8",
        gridcolor: "#1b3027",
      },

      yaxis: {
        title: {
          text: "Frequency (Hz)",
          font: {
            color: "#9db5a8",
          },
        },
        color: "#9db5a8",
        gridcolor: "#1b3027",
      },

      margin: {
        l: 70,
        r: 20,
        t: 50,
        b: 60,
      },

      height: 550,

      font: {
        color: "#e8fff2",
      },
    }}

    config={{
      responsive: true,
      displaylogo: false,
    }}

    style={{
      width: "100%",
    }}
  />
)}
               : (
                <Plot
                  data={[
                    {
                      type: "heatmap",
                      z: result.visualization.spectrogram,
                      x: result.visualization.time,
                      y: result.visualization.frequency,
                      colorscale: "Viridis",
                      colorbar: {
                        title: "dB",
                      },
                      hovertemplate:
                        "Time: %{x:.2f}s" +
                        "<br>Frequency: %{y:.0f} Hz" +
                        "<br>Intensity: %{z:.2f} dB" +
                        "<extra></extra>",
                    },
                  ]}
                  layout={{
                    title: {
                      text: "Bird Call Spectrogram",
                      font: {
                        color: "#e8fff2",
                        size: 18,
                      },
                    },

                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",

                    xaxis: {
                      title: "Time (seconds)",
                      color: "#9db5a8",
                      gridcolor: "#1b3027",
                    },

                    yaxis: {
                      title: "Frequency (Hz)",
                      color: "#9db5a8",
                      gridcolor: "#1b3027",
                    },

                    margin: {
                      l: 70,
                      r: 20,
                      t: 50,
                      b: 60,
                    },

                    height: 550,

                    font: {
                      color: "#e8fff2",
                    },
                  }}
                  config={{
                    responsive: true,
                    displaylogo: false,
                  }}
                  style={{
                    width: "100%",
                  }}
                />
              )}

            </div>

            <div className="visualization-help">
              <span>🖱️ Drag to rotate</span>
              <span>🔍 Scroll to zoom</span>
              <span>↻ Double-click to reset</span>
              <span>✦ Hover for acoustic values</span>
            </div>

          </section>
        )}

        <section className="how-it-works">
          <span className="section-label">HOW IT WORKS</span>

          <div className="steps">

            <div className="step">
              <div className="step-number">01</div>
              <div>
                <h3>Upload</h3>
                <p>
                  Upload a recording of a bird call or song.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">02</div>
              <div>
                <h3>Analyze</h3>
                <p>
                  BirdSense extracts acoustic features using MFCCs.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">03</div>
              <div>
                <h3>Identify</h3>
                <p>
                  Our CNN model predicts the most likely species.
                </p>
              </div>
            </div>

          </div>
        </section>

      </main>
      
      <footer>
        <span>BirdSense-AI</span>
        <span>Real-time acoustic intelligence for birds</span>
      </footer>
    </div>
  );
}

export default App;