"""
証明書の自動設定モジュール。

このPCでは Avast アンチウイルスが HTTPS通信を検査するために
独自の証明書(Avast Web/Mail Shield Root)で通信を中継しています。
Python はこの証明書を標準では信頼しないため、何もしないと
Yahoo! Finance からの株価ダウンロードが全て失敗します。

このモジュールを「他のどのモジュールより先に」import するだけで、
同じフォルダにある ca_bundle.pem (標準証明書 + Avast証明書) を
通信ライブラリに使わせるよう自動設定します。

使い方:  各スクリプトの先頭で
    import certfix  # noqa
と書くだけ。
"""
import os
import sys

# Windowsの標準文字コード(cp932)では絵文字(🔔など)を出力できずクラッシュする。
# 出力を必ずUTF-8にし、万一変換できない文字も「?」に置換して落ちないようにする。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.join(_HERE, "ca_bundle.pem")

if os.path.exists(_BUNDLE):
    # 主要な通信ライブラリ(curl_cffi / requests / 標準ssl)が
    # 参照する環境変数を全て指しておく。
    for var in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        os.environ[var] = _BUNDLE
elif sys.platform == "win32":
    # 証明書ファイルが無い場合は警告だけ出す(止めはしない)。
    # ※Windows以外(GitHub Actions等)ではAvastが存在しないため、この設定自体が不要。
    print(f"[certfix] 警告: 証明書ファイルが見つかりません: {_BUNDLE}")
    print("[certfix] データ取得に失敗する場合は管理者に連絡してください。")
