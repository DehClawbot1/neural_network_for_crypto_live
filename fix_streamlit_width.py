from pathlib import Path


def main() -> None:
    changed = []
    for path in Path('.').rglob('*.py'):
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding='utf-8-sig')
            except Exception:
                continue
        except Exception:
            continue

        new_text = text.replace('use_container_width=True', 'use_container_width=True')
        if new_text != text:
            path.write_text(new_text, encoding='utf-8')
            changed.append(str(path))

    print('Changed files:')
    for item in changed:
        print(f' - {item}')
    print(f'Total changed: {len(changed)}')


if __name__ == '__main__':
    main()
