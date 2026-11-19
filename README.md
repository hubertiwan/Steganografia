# Steganografia – znaki wodne w tekstach generowanych przez LLM

Projekt inżynierski: implementacje pięciu algorytmów znakowania wodnego tekstu tworzonego przez
duże modele językowe oraz aplikacja okienkowa do ich demonstracji, testowania i porównywania.

Znak wodny jest dodawany **w trakcie generowania tekstu** (zmiana logitów albo sposobu
losowania tokenów) i nie widać go w treści. Właściciel klucza wykrywa go testem
statystycznym, bez dostępu do modelu.

| Algorytm | Publikacja | Co ukrywa | Opis |
|---|---|---|---|
| Unigram-Watermark | Zhao i in., ICLR 2024 | 1 bit (czy jest znak) | [docs/unigram.md](docs/unigram.md) |
| SynthID-Text | Dathathri i in., Nature 2024 | 1 bit | [docs/synthid.md](docs/synthid.md) |
| KGW | Kirchenbauer i in., ICML 2023 | 1 bit | [docs/kgw.md](docs/kgw.md) |
| DiPmark | Wu i in., ICML 2024 | 1 bit | [docs/dipmark.md](docs/dipmark.md) |
| MPAC | Yoo i in., NAACL 2024 | wiadomość b-bitowa (np. ID użytkownika) | [docs/mpac.md](docs/mpac.md) |

## Struktura

```
watermarks/          algorytmy, każdy w osobnym pakiecie:
    unigram/  synthid/  kgw/  dipmark/  mpac/
        core.py          klasa znaku wodnego (osadzanie i detekcja)
        generation.py    generowanie tekstu modelem z Hugging Face
demos/               skrypty z eksperymentami w konsoli (jeden na algorytm)
app/                 aplikacja okienkowa (logika), app/gui/ – interfejs PySide6
docs/                opisy algorytmów
tests/               testy algorytmów i aplikacji
main.py              uruchomienie aplikacji
```

## Instalacja

Potrzebny jest Python 3.10 lub nowszy.

```bash
git clone https://github.com/hubertiwan/Steganografia.git
cd Steganografia
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Przy karcie NVIDIA warto najpierw zainstalować PyTorch z obsługą CUDA (instrukcja na pytorch.org).

## Aplikacja

```bash
python main.py
```

Po starcie aplikacja w tle wczytuje model `Qwen/Qwen2.5-0.5B-Instruct` (ten sam co w demach).
Za pierwszym razem jest on pobierany z huggingface.co (ok. 1 GB), potem jest już na dysku.
Inny model albo folder z pobranym modelem można wskazać w menu **Plik → Wczytaj model**.

Zakładki:

- **Algorytmy** – opis każdego algorytmu, parametry, notatka z `docs/` i przypisane dokumenty PDF.
- **Demonstracja** – jedno polecenie: tekst ze znakiem i bez znaku obok siebie, tokeny pokolorowane
  według detektora, wynik detekcji (też z innym kluczem), dla MPAC odczytana wiadomość. Tekst ze
  znakiem można zmienić ręcznie albo losowo i sprawdzić jeszcze raz.
- **Sprawdź tekst** – detekcja w dowolnym wklejonym tekście.
- **Test** – seria poleceń (20 wbudowanych albo z pliku .txt): TPR, FPR, AUC, TPR przy stałym FPR,
  perplexity, czasy, odporność na losowe zmiany tokenów, dla MPAC dokładność bitowa. Wykresy i
  tabela tekstów.
- **Porównanie** – kilka algorytmów na tych samych poleceniach, ziarnach i długości tekstu.
- **Historia** – wszystkie uruchomienia (także przerwane i z błędem) z parametrami i wynikami.
- **Dokumenty PDF** – import publikacji, podgląd stron, wyszukiwanie, przypisanie do algorytmu.

Każdy wynik można zapisać jako PDF (raport z tabelami i wykresami), CSV albo JSON. Historia i
zaimportowane PDF-y są w folderze `app_data/` (nie trafia do gita).

Długie obliczenia działają w osobnym wątku, więc okno nie zamarza, a test można przerwać.
Na procesorze generowanie 100 tokenów modelem 0.5B trwa kilka sekund, więc test na 20 poleceniach
to kilka minut.

## Testy

```bash
pytest -q
```

Testy algorytmów nie potrzebują modelu (sztuczne logity). Testy aplikacji używają małego
modelu testowego i małego GPT-2 budowanego lokalnie, więc też nic nie pobierają.

## Dema w konsoli

```bash
python -m demos.demo_unigram
python -m demos.demo_synthid
python -m demos.demo_kgw
python -m demos.demo_dipmark
python -m demos.demo_mpac
python -m demos.demo_kgw --num-prompts 5 --max-new-tokens 100   # szybciej
```

Każde demo wypisuje wyniki detekcji, perplexity i odporność na zmiany, a wykres zapisuje w `results/`.

## Uwagi

- Perplexity liczy ten sam model, który wygenerował tekst, więc to tylko zgrubna ocena jakości.
- W MPAC decyzja o wykryciu opiera się na dokładnej p-wartości, a nie na progu dla z ze wzoru
  z pracy, bo to z jest zawyżone także dla tekstu bez znaku (szczegóły w [docs/mpac.md](docs/mpac.md)).
- Każdy algorytm ma domyślne parametry ze swojej publikacji, więc progi detekcji i sposób
  próbkowania się różnią. W porównaniu skuteczność najlepiej oceniać przez AUC i TPR przy tym samym FPR.
- Nie są odtworzone ataki parafrazą (wymagają dodatkowych modeli).
