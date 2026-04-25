export interface Book {
  id: string;
  title: string;
  edition: string;
  category: string;
  authors: string;
  citations: number;
  editorialCount: number;
  editorialArea: string;

  year?: string;
  editorial?: string;
  doi?: string;
  abstract?: string;
  keywords?: string;
  language?: string;
  institution?: string;
  matchMethod?: string;
  openAlexId?: string;
  editorialScore?: number;
  citationScore?: number;
}