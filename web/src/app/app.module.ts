import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './components/app.component';
import { HeaderComponent } from './components/header/header.component';
import { ClassificationComponent } from './components/classification/classification.component';
import { HttpClientModule } from '@angular/common/http';
import { IterateGenresPipe } from './pipes/iterateGenres.pipe';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    ClassificationComponent,
    IterateGenresPipe,
  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpClientModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
