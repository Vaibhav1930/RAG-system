import { SignInButton, SignUpButton, Show, UserButton } from '@clerk/nextjs'
import { currentUser } from '@clerk/nextjs/server'

export const dynamic = 'force-dynamic';

export default async function Home() {
  const user = await currentUser();
  const username = user ? (user.username || user.id) : "";

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-white font-sans selection:bg-blue-600 selection:text-white">
      {/* Navigation Bar */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-zinc-950/70 border-b border-zinc-900 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-400 to-pink-500 bg-clip-text text-transparent">
            🧬 SciRAG Workspaces
          </span>
        </div>

        <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-400">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="https://clerk.com/docs" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Docs</a>
        </nav>

        <div className="flex items-center gap-4">
          <Show when="signed-out">
            <SignInButton mode="modal">
              <button className="text-sm font-semibold text-zinc-300 hover:text-white transition-colors cursor-pointer bg-transparent border-0">
                Sign In
              </button>
            </SignInButton>
            <SignUpButton mode="modal">
              <button className="text-sm font-semibold bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg transition-all cursor-pointer border-0">
                Register
              </button>
            </SignUpButton>
          </Show>

          <Show when="signed-in">
            <div className="flex items-center gap-4">
              <span className="text-sm text-zinc-400 hidden sm:inline">
                Workspace: <strong className="text-zinc-200">{username}</strong>
              </span>
              <UserButton />
            </div>
          </Show>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col">
        <section className="relative px-6 py-24 md:py-32 flex flex-col items-center text-center max-w-4xl mx-auto">
          {/* Decorative background glow */}
          <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -top-20 left-1/3 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/20 bg-blue-500/5 text-xs text-blue-400 font-semibold mb-6 animate-pulse">
            <span>✨</span> Powered by Groq Llama 3.3
          </div>

          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight leading-none mb-6">
            Secure, Isolated{" "}
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-pink-500 bg-clip-text text-transparent">
              RAG Workspaces
            </span>{" "}
            for Science
          </h1>

          <p className="text-lg sm:text-xl text-zinc-400 leading-relaxed mb-10 max-w-2xl">
            Upload your technical papers, preserve complex LaTeX equations, and query your secure workspace data with strict user isolation.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center w-full max-w-sm">
            <Show when="signed-out">
              <SignInButton mode="modal">
                <button className="w-full sm:w-auto flex h-12 px-8 items-center justify-center rounded-lg bg-blue-600 font-semibold text-white transition-all hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-600/20 cursor-pointer border-0">
                  Get Started Free
                </button>
              </SignInButton>
            </Show>

            <Show when="signed-in">
              <a 
                href={`${process.env.NEXT_PUBLIC_WORKSPACE_URL || 'http://localhost:8501'}/?username=${encodeURIComponent(username)}`}
                className="w-full sm:w-auto flex h-12 px-8 items-center justify-center rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 font-semibold text-white transition-all hover:opacity-90 shadow-lg shadow-blue-600/25 cursor-pointer text-center leading-12 decoration-0"
              >
                Open Workspace App
              </a>
            </Show>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="border-t border-zinc-900 bg-zinc-950 px-6 py-20">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-center mb-12">
              Engineered for Scientific Inquiry
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Feature 1 */}
              <div className="p-6 rounded-xl border border-zinc-900 bg-zinc-900/30 backdrop-blur-md">
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400 font-bold mb-4">
                  🔒
                </div>
                <h3 className="text-lg font-semibold mb-2">Isolated Workspaces</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  Every user workspace gets its own sanitized Chroma database collection and directory structure on disk. Your documents stay yours.
                </p>
              </div>

              {/* Feature 2 */}
              <div className="p-6 rounded-xl border border-zinc-900 bg-zinc-900/30 backdrop-blur-md">
                <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400 font-bold mb-4">
                  📐
                </div>
                <h3 className="text-lg font-semibold mb-2">Formula Aware Chunking</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  Preserves LaTeX mathematical formatting and scientific formulas ($E=mc^2$) across PDF chunks to maintain precise citations.
                </p>
              </div>

              {/* Feature 3 */}
              <div className="p-6 rounded-xl border border-zinc-900 bg-zinc-900/30 backdrop-blur-md">
                <div className="w-10 h-10 rounded-lg bg-pink-500/10 flex items-center justify-center text-pink-400 font-bold mb-4">
                  ⚡
                </div>
                <h3 className="text-lg font-semibold mb-2">Groq-Speed Latency</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  Leverages Groq's high-speed inference engine running Llama 3.3 models for instantaneous context generation and reasoning.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900 px-6 py-8 text-center text-xs text-zinc-500">
        © {new Date().getFullYear()} SciRAG Workspaces. All rights reserved.
      </footer>
    </div>
  );
}
