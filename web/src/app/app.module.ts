import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { ToastrModule } from 'ngx-toastr';

import { AppComponent } from './components/app.component';
import { HeaderComponent } from './components/header/header.component';
import { ClassificationComponent } from './components/classification/classification.component';
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
    BrowserAnimationsModule,
    ToastrModule.forRoot(),
    FormsModule,
    HttpClientModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
