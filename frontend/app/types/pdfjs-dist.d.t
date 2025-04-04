// src/types/pdfjs-dist.d.ts
declare module "pdfjs-dist/build/pdf" {
  export * from "pdfjs-dist/types/src/pdf";
  const pdfjsLib: any;
  export default pdfjsLib;
}

declare module "pdfjs-dist/build/pdf.worker.entry" {
  const workerSrc: string;
  export default workerSrc;
}
