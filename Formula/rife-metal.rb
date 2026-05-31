class RifeMetal < Formula
  desc "Native macOS RIFE frame interpolation CLI"
  homepage "https://github.com/cinemore/rife-metal"
  license "Apache-2.0"

  depends_on macos: :ventura

  if Hardware::CPU.arm?
    url "https://github.com/cinemore/rife-metal/releases/download/v0.1.1/rife-metal-macos-arm64.tar.gz"
    sha256 "058bed6b9f983c6c2f89f23ce0e44fb22cfe8f632cf5a26598e0438638650247"
  else
    url "https://github.com/cinemore/rife-metal/releases/download/v0.1.1/rife-metal-macos-x86_64.tar.gz"
    sha256 "0000000000000000000000000000000000000000000000000000000000000000"
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
