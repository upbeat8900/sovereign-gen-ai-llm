import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import App from "./App";
import "./styles.css";

const VizPage = lazy(() => import("./viz/VizPage"));

function VizRoute() {
  const { vizId } = useParams();
  if (!vizId) {
    return <Navigate to="/" replace />;
  }
  return <VizPage vizId={vizId} />;
}

export default function RouterApp() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/viz/:vizId"
          element={
            <Suspense
              fallback={
                <div style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>Loading visualization…</div>
              }
            >
              <VizRoute />
            </Suspense>
          }
        />
        <Route path="/*" element={<App />} />
      </Routes>
    </BrowserRouter>
  );
}
