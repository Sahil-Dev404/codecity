"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, Github, Sparkles } from "lucide-react";

const DEMO_REPOS = [
  { name: "pallets/flask", stars: "68k", risk: "12%" },
  { name: "psf/requests", stars: "52k", risk: "8%" },
  { name: "expressjs/express", stars: "65k", risk: "15%" },
];

export default function LandingPage() {
  const router = useRouter();
  const [repoUrl, setRepoUrl] = useState("");

  function handleAnalyze(e: React.FormEvent) {
    e.preventDefault();
    // Still a placeholder slug — once the real backend's POST /analyze
    // exists (Phase 5), this becomes: await submit, get a job id, redirect
    // to /repo/{job_id} once it completes. For now, go straight to the
    // mock city so the full page is reachable and demoable end to end.
    router.push("/repo/demo");
  }

  function handleDemoClick(repoName: string) {
    router.push(`/repo/${repoName.replace("/", "__")}`);
  }

  return (
    <main className="min-h-screen flex flex-col items-center px-6">
      {/* Nav */}
      <nav className="w-full max-w-5xl flex items-center justify-between py-6">
        <div className="flex items-center gap-2 font-mono text-sm text-foreground-muted">
          <Sparkles className="w-4 h-4 text-accent" />
          codecity
        </div>
        <a
          href="https://github.com"
          className="flex items-center gap-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
        >
          <Github className="w-4 h-4" />
          View source
        </a>
      </nav>

      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-2xl text-center mt-20 mb-12"
      >
        <h1 className="text-3xl md:text-[2.75rem] font-semibold tracking-tight text-foreground leading-tight">
          See your codebase as a city.
        </h1>
        <p className="mt-4 text-lg text-foreground-muted leading-relaxed">
          Paste a GitHub repo. A graph neural network predicts which files
          are likely to break next — and shows you why.
        </p>
      </motion.div>

      {/* URL input */}
      <motion.form
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: "easeOut" }}
        onSubmit={handleAnalyze}
        className="w-full max-w-xl flex items-center gap-2 glass-panel rounded-lg p-2"
      >
        <input
          type="url"
          required
          placeholder="https://github.com/owner/repo"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          className="flex-1 bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-foreground-subtle focus:outline-none font-mono"
        />
        <button
          type="submit"
          className="flex items-center gap-1.5 bg-accent hover:bg-accent-muted text-accent-foreground text-sm font-medium px-4 py-2 rounded-md transition-colors"
        >
          Build city
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </motion.form>

      <p className="mt-3 text-xs text-foreground-subtle">
        Public repos only · analysis takes 1–3 minutes
      </p>

      {/* Demo gallery */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}
        className="w-full max-w-3xl mt-20"
      >
        <h2 className="text-sm font-medium text-foreground-muted mb-4">
          Or explore a precomputed city
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {DEMO_REPOS.map((repo) => (
            <button
              key={repo.name}
              onClick={() => handleDemoClick(repo.name)}
              className="glass-panel rounded-lg p-4 text-left hover:border-accent/50 transition-colors group"
            >
              <div className="font-mono text-sm text-foreground group-hover:text-accent transition-colors">
                {repo.name}
              </div>
              <div className="mt-2 flex items-center gap-3 text-xs text-foreground-subtle">
                <span>★ {repo.stars}</span>
                <span className="text-risk-medium">{repo.risk} flagged</span>
              </div>
            </button>
          ))}
        </div>
      </motion.div>
    </main>
  );
}