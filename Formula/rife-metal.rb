class RifeMetal < Formula
  desc "Native Apple Silicon RIFE frame interpolation CLI"
  homepage "https://github.com/cinemore/rife-metal"
  url "https://github.com/cinemore/rife-metal/releases/download/v0.1.0/rife-metal-macos-universal.tar.gz"
  sha256 "5269c48524ad11b2348205bc7f88a9dd7183b441b634523162dd0f1e5b2f730d"
  license "Apache-2.0"

  depends_on macos: :ventura

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
