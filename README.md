# Neural Riviera

## Opis problemu badawczego

W Metodzie Elementów Skończonych (MES) badany obszar dzielimy na siatkę trójkątów, na której przybliżamy rozwiązanie równania różniczkowego. Tam, gdzie błąd obliczeń jest zbyt duży, trzeba zagęścić siatkę - stosuje się wtedy lokalną $h$-adaptację, czyli dzielenie wybranych trójkątów na mniejsze.

Jeśli jednak podzielimy jeden trójkąt, a jego sąsiada zostawimy w spokoju, na ich wspólnej krawędzi powstanie tzw. **hanging node** (wiszący węzeł). Jest to punkt leżący w połowie boku dużego trójkąta, który dla dwóch mniejszych trójkątów po drugiej stronie jest zwykłym wierzchołkiem. Taka sytuacja psuje ciągłość obliczeń, bo po obu stronach tej samej krawędzi spotykają się elementy o różnym stopniu podziału.

Klasycznym sposobem na naprawę takiej siatki jest **algorytm Rivara (najdłuższej krawędzi)**:

- Jeśli na którejś krawędzi trójkąta pojawi się *hanging node*, trzeba ten trójkąt przeciąć na pół, dzieląc jego **najdłuższą krawędź**.
- Dzięki temu nie tworzą się bardzo wąskie, spłaszczone trójkąty, które psują dokładność symulacji.
- Cały proces działa kaskadowo - rozcięcie najdłuższej krawędzi często tworzy kolejny *hanging node* u sąsiedniego trójkąta. Podziały rozchodzą się więc po siatce dotąd, aż znikną wszystkie wiszące węzły (czyli aż podział dojdzie do brzegu siatki albo dwa sąsiednie trójkąty podzielą swoją wspólną najdłuższą krawędź).

Tradycyjnie algorytm ten zapisuje się za pomocą **gramatyk hipergrafowych**. Każdy przypadek na siatce ma swoją osobną regułę (produkcje P1-P9), a program musi nieustannie przeszukiwać graf, dopasowywać odpowiednie podgrafy i pilnować globalnego licznika wiszących węzłów. Takie podejście jest trudne w implementacji, a samo przeszukiwanie grafów słabo nadaje się do równoległego liczenia na GPU.

## Cel pracy

Celem pracy jest sprawdzenie, czy sieć neuronowa potrafi sama nauczyć się zasad rządzących algorytmem Rivara i zastąpić ręcznie pisane reguły naprawcze gramatyki grafowej (**P3-P9**).

Główne założenia rozwiązania:

- **Podejście lokalne (trójkąt po trójkącie):** Sieć dostaje na wejście pojedynczy trójkąt z co najmniej jednym *hanging node* i decyduje, którą z jego trzech krawędzi należy przeciąć.
- **Rozdzielenie decyzji od samego cięcia:** Sieć działa wyłącznie jako klasyfikator - wskazuje numer krawędzi. Resztą zajmuje się nasz skrypt, który dzieli wskazany bok dokładnie na pół i aktualizuje strukturę siatki.
- **Zakres uczenia:** Model trenujemy tylko na etapie **naprawiania siatki** (czyli usuwania wiszących węzłów odpowiadającego regułom P3-P9). Pomijamy początkowe dzielenie elementów wymuszone błędem obliczeń (P1-P2).

## Specyfikacja wejścia i wyjścia (Input / Output)

Żeby sieć patrząca tylko na jeden trójkąt mogła podjąć dobrą decyzję (w tym wiedzieć, kiedy krawędź leży na brzegu siatki i ma priorytet), wektor wejściowy łączy informacje o geometrii elementu z flagami opisującymi stan jego boków.

### Konwencja indeksowania elementu

Żeby zachować porządek zgodny z zapisem gramatyk grafowych, wierzchołki $V_1, V_2, V_3$ oraz krawędzie $e_1, e_2, e_3$ numerujemy cyklicznie:

- Krawędź $e_1$ łączy wierzchołki $V_1$ i $V_2$.
- Krawędź $e_2$ łączy wierzchołki $V_2$ i $V_3$.
- Krawędź $e_3$ łączy wierzchołki $V_3$ i $V_1$.

### Struktura tensora wejściowego i wyjściowego

| Składowa | Typ danych (PyTorch) | Wymiar (2D / 3D) | Znaczenie w algorytmie |
| :--- | :--- | :--- | :--- |
| **Współrzędne wierzchołków** | `torch.float32` | $6$ lub $9$ wartości | Współrzędne $(x_i, y_i)$ lub $(x_i, y_i, z_i)$ dla $i \in \{1, 2, 3\}$ określające położenie i kształt trójkąta. |
| **Flagi hanging nodes ($H$)** | `torch.float32` | $3$ wartości | Wektor $[h_1, h_2, h_3] \in \{0.0, 1.0\}^3$ mówiący, na których krawędziach leżą wiszące węzły. |
| **Flagi brzegowe ($B$)** | `torch.float32` | $3$ wartości | Wektor $[b_1, b_2, b_3] \in \{0.0, 1.0\}^3$ mówiący, czy krawędź leży na zewnętrznym brzegu siatki (`1.0` = brzeg). |
| **Wyjście modelu (Target $Y$)** | `torch.int64` (`long`) | $1$ etykieta ($3$ klasy) | Numer klasy `0`, `1` lub `2` wskazujący odpowiednio krawędź $e_1$, $e_2$ lub $e_3$ do złamania. |

W przestrzeni 2D na wejście sieci wchodzi jednowymiarowy tensor o długości $12$ (w batchu o kształcie `(batch_size, 12)`), a w przestrzeni 3D - o długości $15$ (`(batch_size, 15)`).

### Przykład przepływu danych (Input -> Output)

Spójrzmy na konkretny trójkąt 2D o wierzchołkach $V_1(0.0, 0.0)$, $V_2(4.0, 0.0)$ oraz $V_3(0.0, 5.0)$. Krawędź $e_3$ leży na zewnętrznym brzegu siatki ($b_3 = 1$), na krawędzi $e_1$ siedzi wiszący węzeł ($h_1 = 1$), a najdłuższym bokiem jest wewnętrzna krawędź $e_2$ o długości $L_2 = \sqrt{4^2 + 5^2} \approx 6.403$.

```text
  V3 (0.0, 5.0)
   |\
   | \  e2 (L2 = 6.403, h2 = 0, b2 = 0)  <-- NAJDŁUŻSZA KRAWĘDŹ (do złamania)
   |  \
e3 |   \   (brzeg: b3 = 1, L3 = 5.0)
   |    \
   |     \
   +---*--+  V2 (4.0, 0.0)
  V1      e1 (L1 = 4.0, h1 = 1 [hanging node *], b1 = 0)
(0.0, 0.0)
```

Zamiana tego trójkąta na tensory oraz oczekiwana decyzja sieci wyglądają następująco:

```text
================================================================================
1. INPUT (X) -> Tensor wejściowy (dtype=torch.float32)
================================================================================
Składowe:
- Współrzędne [x1, y1, x2, y2, x3, y3] : [0.0, 0.0, 4.0, 0.0, 0.0, 5.0]
- Hanging nodes [h1, h2, h3]           : [1.0, 0.0, 0.0]
- Krawędzie brzegowe [b1, b2, b3]      : [0.0, 0.0, 1.0]

Wektor wejściowy (shape=[1, 12]):
  X = tensor([[0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

================================================================================
2. MODEL (Forward Pass)
================================================================================
Surowe logity / rozkład Softmax dla klas [e1, e2, e3]:
  probs = tensor([[0.01, 0.98, 0.01]])

================================================================================
3. OUTPUT (Y) -> Decyzja klasyfikatora (dtype=torch.int64)
================================================================================
Predykcja klasy (argmax):
  Y = tensor([1])  --> Indeks 1 oznacza krawędź e2 (w zapisie 1-based: krawędź 2)

Akcja środowiska wykonawczego:
  Złamanie krawędzi e2 w punkcie środkowym V_new = (2.0, 2.5) (reguła propagacji P5).
```

## Przepływ pracy badawczej (Pipeline)

### Syntetyczna generacja zbioru danych (Oracle)

Generujemy losowe siatki początkowe i lokalnie je zagęszczamy, żeby sztucznie wprowadzić *hanging nodes*. Następnie puszczamy na nich klasyczny algorytm naprawczy Rivara (napisany wprost na współrzędnych, bez gramatyk grafowych), który działa jak wyrocznia i zapisuje zbiór treningowy w postaci par `(stan_trójkąta -> wybrana_krawędź)`.

### Analiza reprezentacji cech (Feature Engineering)

Sprawdzamy wpływ lokalnej normalizacji geometrycznej (centrowania trójkąta względem środka ciężkości i skalowania) oraz augmentacji permutacyjnej na skuteczność uczenia na samych współrzędnych wierzchołków. Przyglądamy się też trudnym przypadkom brzegowym - na przykład trójkątom równobocznym, gdzie dwa boki są równe i decyduje to, czy jeden z nich leży na brzegu.

### Trening modelu

Trenujemy klasyfikator w PyTorchu (z funkcją straty `CrossEntropyLoss`), aż bezbłędnie nauczy się reguł wyboru krawędzi na zbiorze walidacyjnym.

### Walidacja w repair loopie

Odłączamy wyrocznię i oddajemy stery wytrenowanej sieci na zupełnie nowych siatkach testowych. Sprawdzamy, czy pętla odpytująca model doprowadzi siatkę do zera *hanging nodes*, nie zapętli się po drodze i nie zdegeneruje kątów trójkątów.

## Proponowane architektury

W ramach eksperymentów sprawdzamy cztery podejścia - od najprostszej sieci na płaskim wektorze, po modele zaprojektowane tak, aby rozumiały symetrię trójkąta i potrafiły bezpośrednio porównywać jego boki.

### Baseline MLP (z mądrym preprocessingiem)

Najprostszy model na start, który bierze na wejście spłaszczony wektor wszystkich danych o trójkącie (`12` liczb w przestrzeni 2D: 6 współrzędnych wierzchołków, 3 flagi *hanging nodes* oraz 3 flagi brzegowe).

- **Jak to działa:** Zamiast wrzucać surowe współrzędne prosto z układu całej siatki, najpierw przesuwamy każdy trójkąt tak, aby jego środek ciężkości znalazł się w punkcie `(0, 0)`, a następnie skalujemy go do stałego rozmiaru. Dzięki temu sieć nie gubi się, gdy trójkąt leży w zupełnie innym miejscu mapy albo staje się mikroskopijny po kilku podziałach. Tak przygotowany wektor przechodzi przez zwykłe warstwy liniowe (`Linear`), normalizację (`LayerNorm`) oraz funkcje aktywacji (`GELU`), zwracając na końcu 3 liczby oznaczające wybór krawędzi `1`, `2` lub `3`.
- **Plusy i minusy:** Sieć jest błyskawiczna i bardzo prosta w napisaniu, ale jest wrażliwa na kolejność wierzchołków. Jeśli w tym samym trójkącie zamienimy numerację krawędzi miejscami, dla zwykłego MLP to zupełnie nowy przykład, więc w trakcie treningu trzeba dodatkowo mieszać kolejność wierzchołków (augmentacja danych).

### DeepSets / Equivariant Edge Network (w stylu PointNet)

Architektura zainspirowana sieciami do chmur punktów 3D. Zamiast sklejać wszystko w jeden długi ciąg liczb, traktujemy trójkąt jako zbiór **3 osobnych krawędzi** - macierz o kształcie `(3, F)`, gdzie każdy wiersz opisuje jeden bok (współrzędne końców, obecność *hanging node* oraz informację o brzegu, co daje $F = 6$ cech w przestrzeni 2D).

- **Jak to działa:** Najpierw każda z trzech krawędzi przechodzi przez dokładnie tę samą małą sieć, która zamienia jej cechy w wektor ukryty (embedding). Następnie wyciągamy **maksimum po wszystkich trzech krawędziach**, liczone osobno dla każdej cechy. Ten wektor maksimum działa jak wspólna miarka pokazująca rekordowe parametry w całym trójkącie (np. największą długość). Na koniec doklejamy ten globalny wynik do każdej krawędzi z osobna i przepuszczamy przez ostatnią warstwę, która ocenia, czy dany bok jest tym właściwym do złamania.
- **Plusy i minusy:** Operacja `max` jest wbudowana w samą architekturę, co naturalnie pasuje do szukania najdłuższej krawędzi. Co więcej, kolejność podania krawędzi przestaje mieć jakiekolwiek znaczenie - jeśli zamienimy wiersze na wejściu, wynik na wyjściu obróci się dokładnie tak samo.

### Edge-Token Transformer (Self-Attention bez Positional Encoding)

Tutaj trójkąt traktujemy jak krótką sekwencję złożoną z **3 tokenów**, gdzie jednym tokenem jest pojedyncza krawędź `(3, F)`.

- **Jak to działa:** Każda krawędź jest najpierw rzutowana na wektor o stałym rozmiarze (`d_model`), ale **celowo nie dodajemy kodowania pozycyjnego** (Positional Encoding). Dzięki temu model nie przywiązuje się do tego, która krawędź była pierwsza, a która trzecia. Następnie działa mechanizm *Self-Attention* - każda krawędź bezpośrednio „rozmawia” z pozostałymi dwiema i zestawia swoje parametry z resztą trójkąta. Na wyjściu każda z nich dostaje własną ocenę punktową.
- **Plusy i minusy:** Mechanizm uwagi świetnie radzi sobie w trudnych przypadkach brzegowych - na przykład wtedy, gdy dwie krawędzie mają identyczną długość i decyzję trzeba podjąć na podstawie tego, która z nich leży na brzegu siatki.

### Pairwise Relation Network (sieć relacyjna)

Model zaprojektowany bezpośrednio pod logikę algorytmu Rivara, który w praktyce sprowadza się do sprawdzania nierówności parami — krawędź $i$ wygrywa, jeśli jest dłuższa lub równa pozostałym dwóm bokom ($L_i \ge L_j$ dla $j \neq i$), z uwzględnieniem priorytetu brzegu przy remisach.

- **Jak to działa:**
  1. **Kodowanie krawędzi:** Najpierw każdą z trzech krawędzi przepuszczamy przez tę samą małą sieć $\phi$, dostając trzy wektory ukryte: $z_i = \phi(e_i)$ dla $i \in \{1, 2, 3\}$.
  2. **Pojedynki $1$ na $1$:** Zamiast wrzucać wszystko naraz, zestawiamy krawędzie parami. Dla każdej pary $(i, j)$ wrzucamy do współdzielonej sieci relacyjnej $g$ oba wektory oraz ich różnicę:
     $$r_{ij} = g\big([z_i \,\|\, z_j \,\|\, z_i - z_j]\big)$$
     Jawne podanie różnicy $z_i - z_j$ daje sieci bezpośredni sygnał do porównania — działa jak wyuczony odpowiednik sprawdzenia $L_i - L_j$ oraz przewagi flagi brzegowej $b_i - b_j$.
  3. **Zliczenie punktów:** Końcowy wynik (logit) dla krawędzi $i$ to po prostu suma punktów zdobytych w starciach z pozostałymi dwoma bokami:
     $$s_i = \sum_{j \neq i} r_{ij}$$
     Krawędź z najwyższym wynikiem $s_i$ zostaje wybrana do podziału.
- **Plusy i minusy:** Najbardziej dosłowne przełożenie reguł matematycznych na strukturę sieci, co ułatwia jej szybkie nauczenie się porównywania długości i priorytetów brzegowych. Ponieważ dodawanie jest przemienne, suma po $j \neq i$ gwarantuje pełną odporność na kolejność podania krawędzi, a przy zaledwie $3$ bokach mamy tylko $6$ par do policzenia, więc model pozostaje bardzo lekki.

### Podsumowanie kandydatów

| Architektura | Kształt wejścia | Odporność na kolejność krawędzi | Sposób porównywania krawędzi | Rola w projekcie |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline MLP** | `(12,)` | Brak (wymaga mieszania danych) | Pośredni (warstwy gęste) | Punkt odniesienia (*benchmark*) |
| **DeepSets / PointNet** | `(3, 6)` | Pełna (`MaxPool` po cechach) | Porównanie z maksimum trójkąta | Lekki i symetryczny model bazowy |
| **Edge-Token Transformer** | `(3, 6)` | Pełna (brak *Positional Encoding*) | Bezpośredni (*Self-Attention*) | Główny model do trudnych przypadków |
| **Pairwise Relation Net** | `(3, 6)` | Pełna (symetryczna suma par) | Jawne pojedynki 1 na 1 | Model odwzorowujący nierówności |