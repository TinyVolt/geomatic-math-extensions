import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';


const pkg = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'package.json'), 'utf-8'));
const commitHash = process.env.VITE_COMMIT_HASH?.slice(0, 7) || '';
const fileName = commitHash ? `${pkg.name}.${commitHash}.mjs` : `${pkg.name}.mjs`;

function isExtensionDef(val: unknown): boolean {
  return (
    val !== null &&
    typeof val === 'object' &&
    typeof (val as any).keyword === 'string' &&
    typeof (val as any).name === 'string' &&
    Array.isArray((val as any).parameters) &&
    typeof (val as any).outputType === 'string' &&
    typeof (val as any).compute === 'function'
  );
}

export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: () => fileName
    },
    outDir: 'dist',
    emptyOutDir: true
  },
  plugins: [
    {
      name: 'generate-manifest',
      async closeBundle() {
        const manifestPath = path.resolve(__dirname, 'manifest.json');
        if (!fs.existsSync(manifestPath)) return;

        const meta = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

        const bundleUrl = pathToFileURL(path.resolve(__dirname, 'dist', fileName)).href;
        const bundle = await import(bundleUrl);

        const extensions = Object.values(bundle)
          .filter(isExtensionDef)
          .map((ext: any) => ({
            id: ext.keyword,
            name: ext.name,
            keyword: ext.keyword,
            entry: fileName,
            parameters: ext.parameters.map((p: any) => ({
              name: p.argName,
              type: p.type,
              ...(p.defaultValue !== undefined ? { default: p.defaultValue } : {})
            })),
            outputType: ext.outputType
          }));

        if (!fs.existsSync(path.resolve(__dirname, 'dist'))) {
          fs.mkdirSync(path.resolve(__dirname, 'dist'));
        }

        fs.writeFileSync(
          path.resolve(__dirname, 'dist/manifest.json'),
          JSON.stringify({ ...meta, extensions }, null, 2)
        );
      }
    }
  ]

});