# MPTour – wielobitowy znak wodny z turniejem (własne rozwiązanie)

**Podstawa:** połączenie MPAC (K. Yoo, W. Ahn, N. Kwak, NAACL 2024, [docs/mpac.md](mpac.md)) i
SynthID-Text (S. Dathathri i in., Nature 2024, [docs/synthid.md](synthid.md)).
**Kod:** `watermarks/mptour/`, testy `tests/test_mptour.py`, demo `python -m demos.demo_mptour`.

## Po co

MPAC przenosi wiadomość b-bitową, ale osadza ją bonusem δ do logitów (jak KGW). To zmienia
rozkład modelu, więc perplexity rośnie z δ. SynthID zachowuje rozkład (uśredniając po kluczach),
bo wybiera token turniejem, ale jest zerobitowy. MPTour bierze z MPAC alokację pozycji i osobną
„tablicę” dla każdej cyfry, a z SynthID sposób wyboru tokenu.

## Jak to działa

**Przygotowanie wiadomości.** Wiadomość zapisujemy w systemie o podstawie r, co daje
b̃ = ⌈b / log₂ r⌉ cyfr (pozycji). Dla r = 4 osiem bitów `10110010` to cyfry `2 3 0 2`. W odróżnieniu
od MPAC r nie zależy od γ (nie ma tu list kolorów).

**Osadzanie (przy generowaniu każdego tokenu):**

1. Ziarno s = hash(klucz, H poprzednich tokenów), jak w SynthID.
2. Pozycja p = hash("pos", klucz, s) mod b̃, cyfra d = m[p], jak w MPAC.
3. Ziarno cyfry s_d = hash("digit", klucz, s, d). Każda cyfra ma więc własne g-funkcje
   g_l(x) = hash("g", klucz, s_d, x, l) & 1, czyli uczciwą monetę dla każdego tokenu i rundy.
4. Tournament sampling z SynthID: 2^m kandydatów z rozkładu modelu, m rund, wygrywa token
   z większym g_l (remis rozstrzyga moneta).

Powtórzony kontekst jest losowany zwykle i pomijany w detekcji, jak w SynthID.

**Odczyt:** dla każdego tokenu odtwarzamy pozycję p i dla każdej możliwej cyfry d' liczymy
sumę g-wartości po rundach: S[p][d'] += Σ_l g_l(x, s_d'). Odczytana cyfra to argmax S[p].

**Detekcja:** w = Σ_p max S[p]. Pod H₀ każdy licznik S[p][d'] ma rozkład Binomial(n_p·m, ½),
a r liczników jednej pozycji jest niezależnych (różne hashe). Stąd P(max S[p] ≤ a) = F_bin(a)^r
dokładnie, a rozkład w to splot po pozycjach, tak samo jak w MPAC. Wykryty, gdy p-wartość ≤ α.
Pokazywane jest też z = (w − nm/2) / √(nm/4). Ono, jak z w MPAC, jest zawyżone dla tekstu bez
znaku (maksimum z r liczników), więc o wykryciu decyduje p-wartość.
`verify(tokens, message)` sprawdza jedną znaną wiadomość zwykłym z-testem SynthID, bez maksimum.

## Dlaczego rozkład jest zachowany

Dla ustalonej cyfry d g-wartości tokenów są niezależnymi uczciwymi monetami, rundy są od siebie
niezależne, a remis rozstrzyga symetryczna moneta. Wtedy dla pary x ≠ y P(x wygrywa) = ½
(uśredniając po kluczu), więc P(wynik = x) = p(x)² + Σ_{y≠x} 2·p(x)·p(y)·½ = p(x). To samo działa
w każdej kolejnej rundzie i dla każdej cyfry, więc jakość tekstu nie zależy od wiadomości.
Sprawdza to test `test_distribution_is_preserved_on_average_over_keys`.
Zastrzeżenia: chodzi o rozkład po temperaturze i top-k, w oczekiwaniu po kluczu i przy
unikalnym kontekście. Ten sam klucz, prompt i wiadomość użyte wielokrotnie dają spójnie
przesunięty rozkład.

## Siła sygnału

W 1. rundzie E[g zwycięzcy] = ½ + ¼·(1 − Σ_x p(x)²) dla właściwej cyfry, a ½ dla pozostałych.
Lift wynosi najwyżej ¼ na rundę i znika przy niskiej entropii. W dalszych rundach maleje,
bo kandydaci (losowani z top-k) coraz częściej się powtarzają. Na token jest to słabiej niż
w MPAC z δ = 2, więc ta sama wiadomość potrzebuje dłuższego tekstu.

## Parametry (domyślne)

| Parametr | Wartość | Znaczenie |
|---|---|---|
| b | 8 bitów | długość wiadomości |
| r | 4 | podstawa zapisu, b̃ = ⌈b / log₂ r⌉ = 4 pozycje (jak MPAC z γ = 0.25) |
| H | 4 | liczba poprzednich tokenów w ziarnie (jak SynthID) |
| m | 8 | rundy turnieju, 2^m = 256 kandydatów (jak SynthID) |
| α | 0.001 | dopuszczalny odsetek fałszywych alarmów detektora |
| temperatura, top-k | 0.7, 100 | jak SynthID |

## Ograniczenia

- Pozycja bez tokenów jest odczytywana jako 0 (pole `empty_positions`).
- Duże r to mniej pozycji, ale więcej konkurentów w argmax. Optimum r nie zostało zmierzone.
- Nie zaimplementowano ważenia rund (SynthID używa średniej ważonej / detektora bayesowskiego)
  ani kodów korekcyjnych na bitach wiadomości.

## Porównanie z wariantem A (listy kolorów MPAC)

Wariant A (`watermarks/mptour_lists/`, testy `tests/test_mptour_lists.py`) zamiast monety ½ dla
każdej cyfry używa list kolorów MPAC: osobna permutacja słownika na rundę, g = 1, gdy token jest
na liście cyfry. Porównanie: `python -m demos.compare_mptour` (Qwen2.5-0.5B-Instruct, 10 promptów,
200 tokenów, 8 bitów, α = 0.001, perplexity bez znaku 2.28):

| wariant | bit acc | cała wiad. | TPR | FPR | z (znana wiad.) | ppl | acc po 10% edycji | ms/token |
|---|---|---|---|---|---|---|---|---|
| B (hash, r = 4) | 0.938 | 0.70 | 0.80 | 0.00 | 5.19 | 2.46 | 0.800 | 13.5 |
| A (listy, γ = 0.25) | 0.887 | 0.40 | 0.60 | 0.00 | 4.50 | 2.57 | 0.838 | 24.8 |
| A (listy, γ = 0.5) | 0.950 | 0.60 | 0.80 | 0.00 | 5.36 | 2.65 | 0.900 | 23.7 |
| MPAC (δ = 2) | 0.988 | 0.90 | 1.00 | 0.20 | — | 3.60 | 0.950 | 10.5 |

Decyzja: zostaje wariant B. A z γ = 0.25 ma słabszy sygnał (√(γ(1−γ)) = 0.433 zamiast 0.5),
a A z γ = 0.5 daje wyniki zbliżone do B, ale jest prawie dwa razy wolniejszy (m permutacji
słownika na kontekst) i wiąże r z γ. Próbka jest mała (10 tekstów), różnice B wobec A(γ = 0.5) są w granicach szumu.
