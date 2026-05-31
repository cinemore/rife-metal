class RifeMetal < Formula
  desc "Native macOS RIFE frame interpolation CLI"
  homepage "https://github.com/cinemore/rife-metal"
  license "Apache-2.0"

  depends_on macos: :ventura

  if Hardware::CPU.arm?
    url "https://github.com/cinemore/rife-metal/releases/download/v0.1.2/rife-metal-macos-arm64.tar.gz"
    sha256 "4d74d9ca29b59c1654dd327de4765d2e592ec73bf5874c301b286d56cd09d854"
  else
    url "https://github.com/cinemore/rife-metal/releases/download/v0.1.2/rife-metal-macos-x86_64.tar.gz"
    sha256 "ffa794a3d40500234367aeb475bf4606d53b048c92f3533e1a02928486911377"
  end

  def install
    libexec.install "bin/rife-metal"
    pkgshare.install "share/rife-metal/rife-v4.26.rmw"

    (bin/"rife-metal").write <<~SH
      #!/bin/sh
      if [ "$#" -eq 0 ]; then
        exec "#{libexec}/rife-metal"
      fi

      has_model=0
      for arg in "$@"; do
        case "$arg" in
          -m|--model|--model=*)
            has_model=1
            ;;
        esac
      done

      if [ "$has_model" -eq 1 ]; then
        exec "#{libexec}/rife-metal" "$@"
      else
        exec "#{libexec}/rife-metal" "$@" --model "#{pkgshare}/rife-v4.26.rmw"
      fi
    SH
  end

  test do
    assert_match "Native Apple Silicon RIFE frame interpolation", shell_output("#{bin}/rife-metal --help")
  end
end
